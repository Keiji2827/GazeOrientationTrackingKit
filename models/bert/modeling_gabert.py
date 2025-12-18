import torch
import copy
from torch import nn
from torch.nn import functional as F
import torchvision.models as models


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        self.n_frames = args.n_frames
        super(GAZEFROMBODY, self).__init__()
        self.BertLayer = BertLayer(args, bert)
        self.HeadMFLayer = HeadMFLayer(args)
        self.LSTMlayer = GazeLSTM(args)

    def forward(self, images, smpl, mesh_sampler, is_train=False):

        dir, mdir = self.BertLayer(images[:,self.n_frames//2], smpl, mesh_sampler, is_train=True)

        R_mode, S_diag, F_mat = self.HeadMFLayer(images, is_train=True)
        dirs = self.LSTMlayer(dir, R_mode, S_diag, is_train=True)

        if is_train == True:
            return dirs, R_mode, S_diag, F_mat
        if is_train == False:
            return dirs[:,self.n_frames//2,:], S_diag#, pred_vertices, pred_camera

class GazeLSTM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_frames = args.n_frames
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, batch_first=False)

    def forward(self, dir, R_mode, S_diag, is_train=False):
        """
        dir: (batch, 3) 基準方向（中心フレームなど）
        R, R_mode: (batch, n_frames-1, 3, 3)
        S_diag: (batch, n_frames-1, 3)
        return: directions: (batch, n_frames, 3)
        """

        dirs = []
        half = self.n_frames//2 # When n_frames is 7, half = 3 
        for i in range(half): # 0,1,2
            x = dir.unsqueeze(-1)  # (batch, 3, 1)
            for j in range(half-1, half -1 - i -1, -1):
                x = torch.matmul(R_mode[:,j].transpose(1,2), x)  # (batch, 3, 1)
            dirs.append(x.squeeze(-1))  # (batch, 3)

        dirs.append(dir)  # (batch, 3)

        for i in range(half+1, self.n_frames):
            x = dir.unsqueeze(-1)  # (batch, 3, 1)
            for j in range(half, i):
                x = torch.matmul(R_mode[:,j], x)  # (batch, 3, 1)
            dirs.append(x.squeeze(-1))  # (batch, 3)

        # dirs をテンソル化 (batch, n_frames, 3)
        dirs = torch.stack(dirs, dim=1)

        dirs_out = dirs.clone()

        # LSTM 入力形式に変換 (n_frames, batch, 3)
        dirs = dirs.transpose(0, 1)

        # LSTM 適用
        lstm_out, _ = self.lstm(dirs)  # (n_frames, batch, 3)

        # 出力を (batch, n_frames, 3) に戻す
        lstm_out = lstm_out.transpose(0, 1)

        dirs_out[:, half, :] = lstm_out[:, half, :]

        return dirs_out


# --------------------------------------------
# EfficientNet-B0 Shallow Feature Extractor
# --------------------------------------------
class EfficientNetShallow(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # 最初の浅い block のみ使用（最軽量）
        # features[:4] → stride 8 程度
        self.features = nn.Sequential(*list(net.features[:4]))
        self.out_channels = 40  # stage 3 output channels (EfficientNet-B0 spec)

    def forward(self, x):

        return self.features(x)  # (B, C=40, H/8, W/8)


# --------------------------------------------
# Correlation Volume (lightweight)
# f1, f2: (B, C, H, W)
# return: corr (B, 1, H, W)
# --------------------------------------------
def correlation_volume(f1, f2):
    corr = (f1 * f2).sum(dim=1, keepdim=True)
    return corr

# --------------------------------------------
# Quaternion → Rotation Matrix
# --------------------------------------------
def quaternion_to_rotation_matrix(q):
    q = q.reshape(q.size(0), 4)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]

    B = q.shape[0]
    R = q.new_zeros((B, 3, 3))

    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)

    return R

# --------------------------------------------
# MLP Head for Quaternion Regression
# --------------------------------------------
class RotationMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), out_dim=4):

        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Linear(last, out_dim))  # 出力次元を指定
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        q = self.mlp(x)
        if q.shape[1] == 4:  # quaternion の場合のみ正規化
            q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return q


class HeadMFLayer(torch.nn.Module):
    def __init__(self, args):
        super(HeadMFLayer, self).__init__()
        self.n_frames = args.n_frames
        self.encoder = EfficientNetShallow()
        self.h = 28
        self.w = 28
        # Flatten correlation volume → MLP
        self.mlp_quat = RotationMLP(in_dim=self.h * self.w, out_dim=4)  # R_mode 用
        self.mlp_S = RotationMLP(in_dim=self.h * self.w, out_dim=3)     # 精度パラメータ S 用


    def forward(self, images, is_train=False):
        #n_batch, n_channel, h, w = image.shape
        #image = image.reshape(n_batch, n_channel, h, w)
        features = []
        # extract low-level features
        #for i in range(self.n_frames):
            #R = self.HeadMFLayer(images[:,i], is_train=True)
        #    feature = self.feature_extractor(images[:,i])
        #    features.append(feature)

        batch_size = images.size(0)
        R_mode_list = []
        S_diag_list = []
        F_list = []

        for i in range(self.n_frames-1):

            feat1 = self.encoder(images[:,i])  # (batch, C, H, W)
            feat2 = self.encoder(images[:,i+1])  # (batch, C, H, W)
            corr = correlation_volume(feat1, feat2)  # (batch, 1, H, W)
            feat = corr.view(corr.size(0), -1)  # (batch, H*W)

            # --- R_mode 推定 ---
            q_mode = self.mlp_quat(feat)  # (batch, 4)
            q_mode = q_mode.squeeze(-1)  # (batch, 4)
            q_mode = q_mode / q_mode.norm(dim=1, keepdim=True).clamp(min=1e-8)
            R_mode = quaternion_to_rotation_matrix(q_mode)
            R_mode_list.append(R_mode)

            # --- 精度パラメータ S ---
            S_diag = F.softplus(self.mlp_S(feat))  # (batch, 3)
            S_diag_list.append(S_diag)

            # --- F 行列の構築 ---
            F_mat = R_mode @ torch.diag_embed(S_diag)
            F_list.append(F_mat)

        R_mode = torch.stack(R_mode_list, dim=1)
        S_diag = torch.stack(S_diag_list, dim=1)
        F_mat = torch.stack(F_list, dim=1)

        return R_mode, S_diag, F_mat

class BertLayer(torch.nn.Module):
    def __init__(self, args, bert):
        super(BertLayer, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(3*14,32)
        self.tanh = torch.nn.PReLU()
        self.encoder2 = torch.nn.Linear(32,3)
        self.encoder3 = torch.nn.Linear(3*14,32)
        self.encoder4 = torch.nn.Linear(32,3)
        #self.encoder3 = torch.nn.Linear(3*90,1)
        self.flatten  = torch.nn.Flatten()
        self.flatten2  = torch.nn.Flatten()

        self.metromodule = copy.deepcopy(bert)
        self.body_mlp1 = torch.nn.Linear(14*3,32)
        self.body_tanh1 = torch.nn.PReLU()
        self.body_mlp2 = torch.nn.Linear(32,32)
        self.body_tanh2 = torch.nn.PReLU()
        self.body_mlp3 = torch.nn.Linear(32,3)

        self.total_mlp1 = torch.nn.Linear(3*2,3*2)
        self.total_tanh1 = torch.nn.PReLU()
        self.total_mlp2 = torch.nn.Linear(3*2,3)

    def transform_head(self, pred_3d_joints):
        Nose = 13

        pred_head = pred_3d_joints[:, Nose,:]
        return pred_3d_joints - pred_head[:, None, :]

    def transform_body(self, pred_3d_joints):
        Torso = 12

        pred_torso = pred_3d_joints[:, Torso,:]
        return pred_3d_joints - pred_torso[:, None, :]


    def forward(self, images, smpl, mesh_sampler, is_train=False):
        batch_size = images.size(0)
        self.bert.eval()
        self.metromodule.eval()

        with torch.no_grad():
            _, tmp_joints, _, _, _, _, _, _ = self.metromodule(images, smpl, mesh_sampler)

        #pred_joints = torch.stack(pred_joints, dim=3)
        pred_joints = self.transform_head(tmp_joints)
        mx = self.flatten(pred_joints)
        mx = self.body_mlp1(mx)
        mx = self.body_tanh1(mx)
        mx = self.body_mlp2(mx)
        mx = self.body_tanh2(mx)
        mx = self.body_mlp3(mx)
        mdir = mx

        # metro inference
        pred_camera, pred_3d_joints, _, _, _, _, _, _ = self.bert(images, smpl, mesh_sampler)

        pred_3d_joints_gaze = self.transform_head(pred_3d_joints)
        x = self.flatten(pred_3d_joints_gaze)
        x = self.encoder1(x)
        x = self.tanh(x)
        x = self.encoder2(x)# [batch, 3]

        dir = self.total_mlp1(torch.cat((x, mdir), dim=1))
        dir = self.total_tanh1(dir)
        dir = self.total_mlp2(dir)


        #dir = dir + mdir#/l2[:,None]

        if is_train == True:
            return dir, mdir
        if is_train == False:
            return dir#, pred_vertices, pred_camera
