


import argparse
from asyncio.log import logger
import os
import random,datetime
import time
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
#import cv2
from models.smpl._smpl import SMPL, Mesh
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.dataloader.gafa_loader import create_gafa_dataset
from models.bert.modeling_gabert import GAZEFROMBODY
from models.utils.geometric_layers import rotation_matrices_from_gaze,rotation_from_two_vectors
from models.utils.matrix_operation_layer import svd_decompose_rotations
from models.utils.matrix_fisher_loss import GazeMFGaussianLoss, MatrixFisherKLLoss
from models.utils.Angle_Error_loss import CosLoss, CosLossSingle
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import save_checkpoint, load_from_state_dict, create_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', 
                        type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', 
                        type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--model_checkpoint", default='output/checkpoint-16-90588/state_dict.bin', 
                        type=str, required=False,
                        help="Path to wholebodygaze checkpoint for inference.")
    parser.add_argument("--model_metro_checkpoint", default='models/weights/metro/metro_for_gaze.pth', 
                        type=str, required=False,
                        help="Path to metro all checkpoint.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--num_train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--lr', "--learning_rate", default=1e-5, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_init_epoch", default=0, type=int, 
                        help="initial epoch number.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument('--logging_steps', type=int, default=10, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument('--is_GAFA', action='store_true', default=False,
                        help="use GAFA dataset or not, default is False")


    args = parser.parse_args()
    return args

def main(args):

    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()
    smpl.eval()

    if 1:
        _metro_network = load_from_state_dict(args, smpl, mesh_sampler)
    else:
        _metro_network = torch.load(args.model_metro_checkpoint, weights_only=False)

    _metro_network.to(args.device)

    _gaze_network = GAZEFROMBODY(args, _metro_network)
    _gaze_network.to(args.device)

    if not args.num_init_epoch == 0:
        state_dict = torch.load(args.model_checkpoint)
        _gaze_network.load_state_dict(state_dict)
        del state_dict

    if not args.test:
        print("Train mode")
        dset = create_dataset(args)
        train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
        train_dset, val_dset = random_split(dset, [len(train_idx), len(val_idx)])

        train_dataloader = DataLoader(
            train_dset, batch_size=1, num_workers=1, pin_memory=True, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dset, batch_size=5, shuffle=False, num_workers=16, pin_memory=True
        )
        # Training
        train(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler)



    else: 
        print("Load wholebodygaze checkpoint from {}".format(args.model_checkpoint))
        state_dict = torch.load(args.model_checkpoint)
        _gaze_network.load_state_dict(state_dict)
        del state_dict

        validate(args, val_dataloader, _gaze_network)

    return 0



def train(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler):
    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)


    frame = args.n_frames // 2
    epochs = args.num_train_epochs
    optimizer = torch.optim.AdamW(
        #params=list(_gaze_network.parameters()),lr=args.lr, 
        [
        {"params": _gaze_network.BertLayer.parameters(), "lr": args.lr * 0.1 * 0.1},
        {"params": _gaze_network.HeadMFLayer.parameters(), "lr": args.lr * 0.1 * 0.1},
        {"params": _gaze_network.LSTMlayer.parameters(), "lr": args.lr}
    ],
        betas=(0.9, 0.999), weight_decay=0
    )

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_cos = AverageMeter()
    log_dir = AverageMeter()
    log_MF = AverageMeter()

    criterion_cos = CosLoss().cuda(args.device)
    criterion_dir = CosLossSingle().cuda(args.device)
    criterion_MF  = MatrixFisherKLLoss().cuda(args.device)

    for epoch in range(args.num_init_epoch, epochs):
        for iteration, batch in enumerate(train_dataloader):

            iteration += 1
            _gaze_network.train()


            batch_imgs = batch['image'].cuda(args.device)
            gaze_dir = batch['gaze_dir'].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)
            head_mask = batch["head_mask"].cuda(args.device)

            batch_size = batch_imgs.size(0)

            #for param_group in optimizer.param_groups:
            #    param_group["lr"] = args.lr

            data_time.update(time.time() - end)

            # forward-pass
            directions, R = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=True)
            #print("R :", R[0])
            #print()

            # SVD 
            pred_F, pred_U, pred_S, pred_V, mode = svd_decompose_rotations(R)
            
            # compute target rotation matrices from gaze directions
            R_target = rotation_from_two_vectors(gaze_dir)
            #print(R[0][0], R_target[0][0])

            print("NaN in R:", torch.isnan(R).any())
            print("NaN in pred_F:", torch.isnan(pred_F).any())
            print("NaN in pred_U:", torch.isnan(pred_U).any())
            print("NaN in pred_S:", torch.isnan(pred_S).any())
            print("NaN in pred_V:", torch.isnan(pred_V).any())
            print("NaN in R_target:", torch.isnan(R_target).any())

            # loss
            loss_cos = criterion_cos(directions, gaze_dir).mean()
            loss_dir = criterion_dir(directions[:,frame,:],gaze_dir[:,frame,:]).mean()
            loss_MF  = criterion_MF(pred_F, pred_S, R_target).mean()

            print(loss_MF)

            a = 0.5
            b = 0.5
            loss = (a)*loss_cos  + b*loss_dir  + loss_MF

            # update logs
            log_losses.update(loss.item(), batch_size)
            log_cos.update(loss_cos.item(), batch_size)
            log_dir.update(loss_dir.item(), batch_size)
            log_MF.update(loss_MF.item(), batch_size)

            # back prop
            optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(_gaze_network.parameters(), max_norm=1.0)
            optimizer.step()


            for name, p in _gaze_network.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        print("NaN grad in", name)
                        break
            batch_time.update(time.time() - end)
            end = time.time()


            if iteration%args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(f"eta: {eta_string}, epoch: {epoch}, iter: {iteration},"
                    f" loss: {log_losses.avg:.4f}, log_dir: {log_dir.avg:.2f}, log_MF: {log_MF.avg:.2f}")


        checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
        print("save trained model at ", checkpoint_dir)


        val = validate(args, val_dataloader, 
                            _gaze_network, 
                            smpl, 
                            mesh_sampler
                )
        print("val:", val)


    return 0


def validate(args, val_dataloader, gaze_network, smpl, mesh_sampler):
    max_iter = len(val_dataloader)
    end = time.time()
    batch_time = AverageMeter()

    #mse = AverageMeter()
    log_losses = AverageMeter()

    gaze_network.eval()
    frame = args.n_frames // 2
    criterion = CosLossSingle().cuda(args.device)

    with torch.no_grad():        
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            epoch = iteration

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)
            gaze_dir = gaze_dir[:,frame,:]

            # forward-pass
            direction = gaze_network(batch_imgs, smpl, mesh_sampler, is_train=False)
            direction = direction[0]
            #print(direction.shape)

            loss = criterion(direction,gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if iteration%args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(f"eta: {eta_string}, epoch: {epoch}, iter: {iteration}, loss: {log_losses.avg:.4f}")

    return log_losses.avg




if __name__ == "__main__":
    args = parse_args()
    main(args)
