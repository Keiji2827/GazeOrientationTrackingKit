


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
from models.utils.metric_logger import AverageMeter

from torchvision import transforms

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

    if 1:
        _metro_network = load_from_state_dict(args)
    else:
        _metro_network = torch.load(args.model_metro_checkpoint, weights_only=False)

    _metro_network.to(args.device)

    _gaze_network = GAZEFROMBODY(args, _metro_network)
    _gaze_network.to(args.device)



    if not args.test:
        print("Train mode")
        dset = create_dataset(args)
        #train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
        #train_dset, val_dset = random_split(dset, [len(train_idx), len(val_idx)])

        split_1 = int(len(dset)*0.9)
        split_2 = int(len(dset)*0.1)
        train_dset, val_dset, test_dset = random_split(dset, [split_1, split_2-split_1, len(dset)-split_2])

        train_dataloader = DataLoader(
            train_dset, batch_size=1, num_workers=16, pin_memory=True, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
        )
        # Training
        train(args, train_dataloader, val_dataloader, _gaze_network)



    else: 
        print("Load wholebodygaze checkpoint from {}".format(args.model_checkpoint))
        state_dict = torch.load(args.model_checkpoint)
        _gaze_network.load_state_dict(state_dict)
        del state_dict

        validate(args, val_dataloader, _gaze_network)

    return 0

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))

    os.makedirs(checkpoint_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, os.path.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir



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
        #cos[cos != cos] = 0
        cos[cos > 999/1000] = 999/1000
        cos[cos < -999/1000] = -999/1000
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss


def train(args, train_dataloader, val_dataloader, _gaze_network):
    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)

    epochs = args.num_train_epochs
    optimizer = torch.optim.Adam(params=list(_gaze_network.parameters()),lr=args.lr, betas=(0.9, 0.999), weight_decay=0)

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_cos = AverageMeter()


    criterion_cos = CosLoss().cuda(args.device)

    for epoch in range(args.num_init_epoch, epochs):
        for iteration, batch in enumerate(train_dataloader):

            iteration += 1
            _gaze_network.train()


            batch_imgs = batch['image'].cuda(args.device)
            gaze_dir = batch['gaze_dir'].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)

            batch_size = batch_imgs.size(0)

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

            data_time.update(time.time() - end)

            # forward-pass
            direction, mdirection = _gaze_network(batch_imgs, is_train=True)

            # loss
            loss_cos = criterion_cos(direction,gaze_dir[:,(args.n_frames-1)//2]).mean()


            a = 0.7
            loss = (a)*loss_cos # + (1-a)*loss_head

            # update logs
            log_losses.update(loss.item(), batch_size)
            log_cos.update(loss_cos.item(), batch_size)

            # back prop
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()


            if iteration%args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(f"eta: {eta_string}, epoch: {epoch}, iter: {iteration},"
                    f" loss: {log_losses.avg:.4f}, cos: {log_cos.avg:.2f}")


        checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
        print("save trained model at ", checkpoint_dir)


        val = validate(args, val_dataloader, 
                            _gaze_network, 
                )
        print("val:", val)


    return 0


def validate(args, val_dataloader, gaze_network):
    max_iter = len(val_dataloader)
    end = time.time()
    batch_time = AverageMeter()

    mse = AverageMeter()

    gaze_network.eval()

    criterion = CosLoss().cuda(args.device)

    with torch.no_grad():        
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            epoch = iteration

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)

            # forward-pass
            direction, _ = gaze_network(batch_imgs)
            #print(direction.shape)

            loss = criterion(direction,gaze_dir).mean()

            # update logs
            mse.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if iteration%args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(f"eta: {eta_string}, epoch: {epoch}, iter: {iteration}")

    return mse.avg




def load_from_state_dict(args):
    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # load pretrained model
    # Build model from scratch, and load weights from state_dict.bin
    trans_encoder = []
    # input_feat_dim default='2051,512,128'
    input_feat_dim = [2051, 512, 128] #[int(item) for item in args.input_feat_dim.split(',')]
    # hidden_feat_dim default='1024,256,128'
    hidden_feat_dim = [1024, 256, 128] #[int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # output_feat_dim default='512,128,3'
    for i in range(len(output_feat_dim)):
        # from metro.modeling.bert import BertConfig, METRO
        config_class, model_class = BertConfig, METRO
        # default='metro/modeling/bert/bert-base-uncased/'
        config = config_class.from_pretrained(args.model_name_or_path)

        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]

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

    hrnet_yaml = 'models/hrnet/weights/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    hrnet_checkpoint = 'models/hrnet/weights/hrnetv2_w64_imagenet_pretrained.pth'
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    #logger.info('=> loading hrnet-v2-w64 model')

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    #total_params = sum(p.numel() for p in trans_encoder.parameters())
    #backbone_total_params = sum(p.numel() for p in backbone.parameters())


    print(type(backbone))

    _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_smpl, mesh_sampler)

    state_dict = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
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


    return _metro_network

def loding_images(args):
    image_list = []

    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if os.path.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif os.path.isdir(args.image_file_or_path):
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_list.append(args.image_file_or_path+'/'+filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))
    return image_list


def create_dataset(args):
    print(f"Creating dataset, is_GAFA: {args.is_GAFA}")
    if not args.is_GAFA:
        exp_names = [
            'data20',
            'data23',
            'data25',
            'data29_0',
            'data29_1',
            'data29_2',
        ]
        random.shuffle(exp_names)
        # Ryukoku dataset
        dset = create_gafa_dataset(exp_names=exp_names, root_dir='data/GoTK', n_frames=args.n_frames)

    if args.is_GAFA:
        exp_names = [
        'living_room/005',
        #'living_room/004',
        #'kitchen/1015_4',
        #'kitchen/1022_4',
        #'library/1028_2',
        #'library/1028_5',
        #'library/1026_3',
        #'courtyard/004',
        #'courtyard/005',
        #'lab/1013_1',
        #'lab/1014_1',
                    ]
        random.shuffle(exp_names)
        # GAFA dataset
        dset = create_gafa_dataset(exp_names=exp_names, n_frames=args.n_frames)

    return dset




if __name__ == "__main__":
    args = parse_args()
    main(args)
