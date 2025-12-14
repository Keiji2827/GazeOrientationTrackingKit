


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
from models.utils.logger import setup_logger
from models.bert.modeling_gabert import GAZEFROMBODY
from models.utils.metric_logger import AverageMeter

from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        l2 = torch.linalg.norm(outputs, ord=2, axis=1)
        outputs = outputs/l2[:,None]
        outputs = outputs.reshape(-1, outputs.shape[-1])

        l2 = torch.linalg.norm(targets, ord=2, axis=1)
        targets = targets/l2[:,None]
        targets = targets.reshape(-1, targets.shape[-1])

        cos =  torch.sum(outputs*targets,dim=-1)
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss

def run_test(args, test_dataloader, _gaze_network, smpl, mesh_sampler):

    print("len of dataset:", len(test_dataloader))
        
    val = run_validate(args, test_dataloader, 
                        _gaze_network, 
                        #criterion_mse,
                        smpl,
                        mesh_sampler
                        )

    print(args.dataset)
    print("test:", val)

def run_validate(args, val_dataloader, _gaze_network, smpl,mesh_sampler):


    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    max_iter = len(val_dataloader)

    _gaze_network.eval()
    smpl.eval()

    frame = args.n_frames // 2

    criterion_mse = CosLoss().cuda(args.device)

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
            direction = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=False)
            # direction: tuple (batch, n_frames, 3)
            #direction = direction[0]

            loss = criterion_mse(direction, gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if(iteration%10==0):
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    ' '.join(
                    ['eta: {eta}', 'iter: {iter}']
                    ).format(eta=eta_string, iter=iteration)
                    + ", loss:{:.4f}".format(log_losses.avg) 
                )

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
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    # default='output/'
    #mkdir(args.output_dir)
    logger = setup_logger("WholeBodyGaze Test", args.output_dir, 0)
    # randomのシード
    # default=88
    #set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    # from metro.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local
    #renderer = Renderer(faces=mesh_smpl.faces.cpu().numpy())

    # Load pretrained model
    # --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # どうやらこっち側みたい
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        # input_feat_dim default='2051,512,128'
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        # hidden_feat_dim default='1024,256,128'
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        # transformerの初期化
        # output_feat_dim = [512, 128, 3]
        for i in range(len(output_feat_dim)):
            # from metro.modeling.bert import BertConfig, METRO
            config_class, model_class = BertConfig, METRO
            # default='metro/modeling/bert/bert-base-uncased/'
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                # During our paper submission, we were using the original intermediate size, which is 3072 fixed
                # We keep our legacy setting here 
                args.intermediate_size = -1
            else:
                # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
                # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
                args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    #logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            # model_class = METRO
            model = model_class(config=config) 
            #logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # for ここまで
        # init ImageNet pre-trained backbone model
        # arch default='hrnet-w64'
        if args.arch=='hrnet':
            hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            #logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/weights/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/weights/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            #logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        # from metro.modeling.bert import METRO_Body_Network as METRO_Network
        # ここでモデルの初期化
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        #logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    # model の構築ここまで

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run Test")

    _gaze_network = GAZEFROMBODY(args, _metro_network)
    _gaze_network.to(args.device)

    if args.device == 'cuda':
        print("distribution")
        _gaze_network = torch.nn.DataParallel(_gaze_network) # make parallel
        torch.backends.cudnn.benchmark = True

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
    #dset = create_gafa_dataset(exp_names=['data20','data23','data25'], root_dir='../MakeDataset', test=True, augumented=False)
    #dset = create_gafa_dataset(exp_names=exp_names, root_dir='data/GoTK', test=True, augumented=False)

    test_dataloader = DataLoader(
        #dset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
        dset, batch_size=72, shuffle=True, num_workers=1, pin_memory=True
    )

    run_test(args, test_dataloader, _gaze_network, mesh_smpl, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
