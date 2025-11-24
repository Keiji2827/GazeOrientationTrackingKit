import torch
import copy
#from torch import nn
#import numpy as np
#from .modeling_bert import BertLayerNorm as LayerNormClass
#from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
#from models.utils.geometric_layers import orthographic_projection


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        self.n_frames = args.n_frames
        super(GAZEFROMBODY, self).__init__()
        self.BertLayer = BertLayer(args, bert)
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=20, batch_first=True)
        self.fc = torch.nn.Linear(20, 3)  # 予測値1つ
        self.lstm2 = torch.nn.LSTM(input_size=3, hidden_size=20, batch_first=True)
        self.fc2 = torch.nn.Linear(20, 3)  # 予測値1つ

    def forward(self, images, is_train=False, render=False):

        direction = []
        mdirection = []
        for i in range(self.n_frames):
            dir, mdir = self.BertLayer(images[:,i], is_train=True)
            direction.append(dir)
            mdirection.append(mdir)

        direction_seq = torch.stack(direction, dim=1)  # shape: [B, n_frames, D]
        mdirection_seq = torch.stack(mdirection, dim=1)  # shape: [B, n_frames, D]

        x , _= self.lstm(direction_seq)  # LSTMの出力を取得
        x = x[:,-1,:]  # 最後のタイムステップの出力を全結合層に通す
        x = self.fc(x)

        mx, _ = self.lstm2(mdirection_seq)  # LSTMの出力を取得
        mx = mx[:,-1,:]  # 最後のタイムステップの outputを全結合層に通す
        mx = self.fc2(mx)

        return x, mx


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


    def forward(self, images, is_train=False, render=False):
        batch_size = images.size(0)
        self.bert.eval()
        self.metromodule.eval()

        with torch.no_grad():
            _, tmp_joints, _, _, _, _, _, _ = self.metromodule(images)

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
        pred_camera, pred_3d_joints, _, _, _, _, _, _ = self.bert(images)
        pred_3d_joints_gaze = self.transform_head(pred_3d_joints)
        x = self.flatten(pred_3d_joints_gaze)
        x = self.encoder1(x)
        x = self.tanh(x)
        x = self.encoder2(x)# [batch, 3]

        dir = self.total_mlp1(torch.cat((x, mx), dim=1))
        dir = self.total_tanh1(dir)
        dir = self.total_mlp2(dir)

        dir = dir + mx#/l2[:,None]


        if is_train == True:
            return dir, mdir
        if is_train == False:
            return dir#, pred_vertices, pred_camera
