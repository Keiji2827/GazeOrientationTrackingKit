


from __future__ import absolute_import, division, print_function
import argparse
import os
import time
import datetime
import torch
#import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.smpl._smpl import SMPL, Mesh
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.dataloader.gafa_loader import create_gafa_dataset
#from models.utils.logger import setup_logger
from models.bert.modeling_gabert import GAZEFROMBODY
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import load_from_state_dict
from models.utils.Angle_Error_loss import CosLossSingle



def run_test(args, test_dataloader, _gaze_network, smpl, mesh_sampler):

    print("len of dataset:", len(test_dataloader))
        
    val = run_validate(args, test_dataloader, 
                        _gaze_network, 
                        #criterion_mse,
                        smpl,
                        mesh_sampler
                        )

    print(args.dataset)
    print("test:", torch.rad2deg(val))

def run_validate(args, val_dataloader, _gaze_network, smpl,mesh_sampler):


    end = time.time()
    batch_time = AverageMeter()

    log_losses = AverageMeter()

    _gaze_network.eval()
    frame = args.n_frames // 2
    criterion = CosLossSingle().cuda(args.device)

    data_time = AverageMeter()
    max_iter = len(val_dataloader)

    smpl.eval()

    with torch.no_grad():
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            epoch = iteration

            batch_imgs = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)

            #batch_imgs = image
            batch_size = batch_imgs.size(0)
            gaze_dir = gaze_dir[:,frame,:]

            data_time.update(time.time() - end)

            # forward-pass
            direction, S_diag = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=False)
            # direction: tuple (batch, n_frames, 3)

            loss = criterion(direction, gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if(iteration%args.logging_steps==0):
                now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"date: {now}, eta: {eta_string}, epoch: {epoch}, iter: {iteration}, loss: {log_losses.avg:.4f}")

    return log_losses.avg


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--model_checkpoint", default='output/checkpoint-6-54572/state_dict.bin', type=str, required=False,
                        help="Path to wholebodygaze checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
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
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument('--logging_steps', type=int, default=10, 
                        help="Log every X steps.")
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument('--dataset', type=str, nargs='*', default="", 
                        help="use test scene.")

    args = parser.parse_args()
    return args

# 最初はここから
def main(args):
    #global logger
    # Setup CUDA, GPU & distributed training
    #args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    #args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    # default='output/'
    #mkdir(args.output_dir)
    #logger = setup_logger("WholeBodyGaze Test", args.output_dir, 0)
    # randomのシード
    # default=88
    #set_seed(args.seed, args.num_gpus)
    #logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()
    smpl.eval()


    # Load pretrained model
    # --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
    #logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    _metro_network = load_from_state_dict(args, smpl, mesh_sampler)
    _gaze_network = GAZEFROMBODY(args, _metro_network)
    _gaze_network.to(args.device)

    #if args.device == 'cuda':
    #    print("distribution")
    #    _gaze_network = torch.nn.DataParallel(_gaze_network) # make parallel
    #    torch.backends.cudnn.benchmark = True

    state_dict = torch.load(args.model_checkpoint)
    _gaze_network.load_state_dict(state_dict)
    del state_dict

    exp_names = [
        'library/1029_2', #
        'lab/1013_2',
        'kitchen/1022_2',
        'living_room/006',
        'courtyard/002',
        'courtyard/003',
    ]

    if args.dataset:
        exp_names = args.dataset

    dset = create_gafa_dataset(exp_names=exp_names, n_frames=args.n_frames)

    test_dataloader = DataLoader(
        #dset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
        dset, batch_size=72, shuffle=True, num_workers=1, pin_memory=True
    )

    run_test(args, test_dataloader, _gaze_network, smpl, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
