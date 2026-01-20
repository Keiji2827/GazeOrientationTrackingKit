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

        R_mode, S_diag, F_mat, d_corr = self.HeadMFLayer(images, is_train=True)
        dirs = self.LSTMlayer(dir, R_mode, S_diag, is_train=True)

        if is_train == True:
            return dirs, mdir, R_mode, S_diag, F_mat, d_corr
        if is_train == False:
            return dirs[:,self.n_frames//2,:], S_diag#, pred_vertices, pred_camera


def cumulative_matmul(R):
    B, N, _, _ = R.shape
    outs = []

    cur = R[:, 0]
    outs.append(cur)

    for i in range(1, N):
        cur = cur @ R[:, i]
        outs.append(cur)

    return torch.stack(outs, dim=1)

class GazeLSTM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_frames = args.n_frames
        # ✅ batch_first=False に戻す（元コードと同じ）
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, batch_first=False)

    def forward(self, dir, R_mode, S_diag, is_train=False):
        """
        dir: (batch, 3)
        R_mode: (batch, n_frames-1, 3, 3)
        """
        B, Tm1, _, _ = R_mode.shape
        T = Tm1 + 1
        half = T // 2

        # ============================
        # forward cumulative rotations
        # ============================
        R_fwd = cumulative_matmul(R_mode[:, half:])  # (B, T-half-1, 3, 3)

        # ============================
        # backward cumulative rotations
        # ============================
        R_bwd = cumulative_matmul(
            R_mode[:, :half].transpose(2,3).flip(1)
        )  # (B, half, 3, 3)

        dirs = []

        # past
        if half > 0:
            d_past = torch.matmul(
                R_bwd,
                dir[:, None, :, None]
            ).squeeze(-1)  # (B, half, 3)
            dirs.append(d_past.flip(1))

        # center
        dirs.append(dir[:, None, :])  # (B, 1, 3)

        # future
        if T - half - 1 > 0:
            d_fut = torch.matmul(
                R_fwd,
                dir[:, None, :, None]
            ).squeeze(-1)  # (B, T-half-1, 3)
            dirs.append(d_fut)

        # (B, T, 3)
        dirs = torch.cat(dirs, dim=1)

        # ============================
        # LSTM（元コードと同じ動作）
        # ============================

        # ✅ (B, T, 3) → (T, B, 3)
        dirs_lstm_in = dirs.transpose(0, 1)

        lstm_out, _ = self.lstm(dirs_lstm_in)  # (T, B, 3)

        # ✅ (T, B, 3) → (B, T, 3)
        lstm_out = lstm_out.transpose(0, 1)

        # ✅ 中心フレームだけ置き換える（元コードと同じ）
        dirs_out = dirs.clone()

        # ✅ in-place を避ける
        center = lstm_out[:, half, :].unsqueeze(1)  # (B,1,3)

        dirs_out = torch.cat([
            dirs_out[:, :half, :],
            center,
            dirs_out[:, half+1:, :]
        ], dim=1)

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

def debug_check_tensors(**tensors):
    for name, t in tensors.items():
        if not torch.isfinite(t).all():
            print(f"Non-finite in {name}: nan={torch.isnan(t).any().item()}, inf={torch.isinf(t).any().item()}, min={t.min().item()}, max={t.max().item()}")


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
        B, T, C, H, W = images.shape
        device = images.device


        debug_check_tensors(images=images)
        #print("images dtype:", images.dtype)
        #print("images min/max:", images.min().item(), images.max().item())

        # ============================================================
        # ✅ 1. EfficientNetShallow をフレームごとに1回だけ実行
        # ============================================================
        images_flat = images.reshape(B * T, C, H, W)
        feats_flat = self.encoder(images_flat)  # (B*T, C', H', W')
        debug_check_tensors(feats_flat=feats_flat)
        C2, H2, W2 = feats_flat.shape[1:]
        feats = feats_flat.reshape(B, T, C2, H2, W2)

        # ============================================================
        # ✅ 2. correlation_volume をベクトル化（ループなし）
        # ============================================================
        feat1 = feats[:, :-1]      # (B, T-1, C2, H2, W2)
        feat2 = feats[:, 1:]       # (B, T-1, C2, H2, W2)
        feat1_n = F.normalize(feat1, p=2, dim=2, eps=1e-8)
        feat2_n = F.normalize(feat2, p=2, dim=2, eps=1e-8)
        corr = (feat1_n * feat2_n).sum(dim=2, keepdim=True)  # in [-1, 1]
        #corr = (feat1 * feat2).sum(dim=2, keepdim=True)  # (B, T-1, 1, H2, W2)
        #corr = torch.clamp(corr, min=-20.0, max=20.0)
        debug_check_tensors(corr=corr)
        corr = corr.reshape(B, T-1, -1)                  # (B, T-1, H2*W2)

        # ============================================================
        # ✅ 3. MLP を一括処理（ループなし）
        # ============================================================
        corr_flat = corr.reshape(B * (T-1), -1)
        debug_check_tensors(corr_flat=corr_flat)


        q_mode = self.mlp_quat(corr_flat)  # (B*(T-1), 4)
        debug_check_tensors(q_mode=q_mode)

        q_mode = q_mode / q_mode.norm(dim=1, keepdim=True).clamp(min=1e-8)
        R_mode = quaternion_to_rotation_matrix(q_mode)
        debug_check_tensors(R_mode=R_mode)

        R_mode = R_mode.reshape(B, T-1, 3, 3)

        # --- S ---------------------------------------------------
        S_diag = F.softplus(self.mlp_S(corr_flat))
        debug_check_tensors(S_diag=S_diag)

        s_max = 8.0                           # ← 推奨上限
        S_diag = torch.clamp(S_diag, max=s_max)
        S_diag = S_diag.reshape(B, T-1, 3)

        # ============================================================
        # ✅ 4. F_mat も一括処理
        # ============================================================
        F_mat = R_mode @ torch.diag_embed(S_diag)

        return R_mode, S_diag, F_mat, corr

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
        #self.bert.train()
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
