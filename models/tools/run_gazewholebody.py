

import argparse
import os
import os.path as op
import time
import datetime
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.bert.modeling_gabert import GAZEFROMBODY
from models.smpl._smpl import SMPL, Mesh
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.dataloader.gafa_loader import create_gafa_dataset
from models.utils.logger import setup_logger
from models.utils.metric_logger import AverageMeter

from PIL import Image
from torchvision import transforms


transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])


class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        cos =  torch.sum(outputs*targets,dim=-1)
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss

class MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        loss = ((outputs-targets)**2).mean()

        return loss

def run(args, train_dataloader, val_dataloader, gaze_model, smpl, mesh_sampler):

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter
    args.logging_steps = 500

    optimizer = torch.optim.Adam(params=list(gaze_model.parameters()),lr=args.lr,
                                            betas=(0.9, 0.999), weight_decay=0) 

    start_training_time = time.time()
    end = time.time()
    gaze_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()

    criterion_mse = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_gaze = CosLoss().cuda(args.device)
    criterion_body = CosLoss().cuda(args.device)
    criterion2 = MSE().cuda(args.device)

    print("length of train_dataloader",len(train_dataloader))   
    for iteration, batch in enumerate(train_dataloader):

        gaze_model.eval()
        iteration += 1
        epoch = iteration
        image = batch["image"]
        gaze_dir = batch["gaze_dir"]
        head_dir = batch["head_dir"]
        body_dir = batch["body_dir"]
        head_pos = batch["head_pos"]
        keypoints = batch["keypoints"]

        #img = Image.open(image)
        # from torchvision import transforms
        #img_tensor = transform(image)
        # 指定した位置にサイズ1の次元を挿入する unsqeeze()
        #batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()        

        #batch_size = image.size()[0]
        #adjust_learning_rate(optimizer, epoch, args)
        # Sets the learning rate to the initial LR decayed by x every y epochs
        #lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        data_time.update(time.time() - end)

        image = image.cuda(args.device)
        gaze_dir = gaze_dir.cuda(args.device)
        head_dir = head_dir.cuda(args.device)
        body_dir = body_dir.cuda(args.device)
        head_pos = head_pos.cuda(args.device)
        keypoints = keypoints.float().cuda(args.device)

        # forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = gaze_model(image, smpl, mesh_sampler)
        #pred = gaze_model(image, smpl, mesh_sampler, is_train=True)
        #print(pred_keypoints.dtype)
        #print(gaze_dir.shape)
        #print(pred_keypoints.shape)
        return
        #
        #print("size of gaze_dir:",gaze_dir[0])
        #print("loss is ", loss_manual)
        # compute loss function
        #print("loss:",loss)
        #loss = 0.5*(loss_gaze.mean()+loss_body.mean())
        #loss0 = criterion_gaze(pred_gaze,gaze_dir).mean()
        #loss1 = criterion_body(pred_gaze,body_dir).mean()
        loss1 = criterion_mse(pred_keypoints,keypoints).mean()
        #loss2 = criterion2(pred_gaze,gaze_dir).mean()
        #print("loss:",loss)
        loss = loss1#(loss0 + loss1)*0.5
        # update logs
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
    
        if(iteration%10==0):
            #print("iteration:",iteration)
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + ":loss:{:.4f}, lr:{:.6f}".format(log_losses.avg, optimizer.param_groups[0]["lr"])
            )
            #print(pred_head_pos[0])
            #print(head_pos[0])
            #print("gaze_dir:",gaze_dir)


#def run_validate(args, val_datalo)
            


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./data/sample', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-3, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
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
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")

    args = parser.parse_args()
    return args

def main(args):

    global logger

    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("model Test", args.output_dir, 0)

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:]+[3]

    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            #config = config_class.from_pretrained(args.config_name if args.config_name else \
            #                                        args.model_name_or_path)
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = -1

            # update model structure if specified in argments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)
        
            # init ImageNet pre-trained backbone model
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            trans_encoder.append(model)
                
        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')

        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/weights/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/weights/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
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

        # Initialize GAZEBERT model 
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

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
    #if args.device == "cuda":
    #    _gaze_bert = torch.nn.DataParallel(_gaze_bert)
    #_gaze_bert = GAZEFROMBODY(args, _metro_network)
    #_gaze_bert.to(args.device)

    logger.info("Training parameters %s", args)

    if args.run_eval_only == True:
        logger.info("Run eval only\nNot use")
    else:
        logger.info("Run train")
        exp_names = ["courtyard/002/","courtyard/003/","courtyard/004/",
                     "kitchen/1015_4","kitchen/1022_2"]
        dset = create_gafa_dataset(exp_names=exp_names)
        train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
        train_dset = Subset(dset, train_idx)
        val_dset   = Subset(dset, val_idx)

        train_dataloader = DataLoader(
            train_dset, batch_size=16, num_workers=4, pin_memory=True, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )
        
        run(args, train_dataloader, val_dataloader, _metro_network, mesh_smpl, mesh_sampler)



if __name__ == "__main__":
    args = parse_args()
    main(args)
