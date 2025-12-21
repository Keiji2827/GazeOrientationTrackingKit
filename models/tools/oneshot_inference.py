"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image

python ./metro/tools/end2end_inference_bodymesh.py 
       --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
       --image_file_or_path ./samples/human-body
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import copy
import torch
import torchvision.models as models
#from torchvision.utils import make_grid
import numpy as np
import cv2
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.bert.modeling_gabert import GAZEFROMBODY
from models.smpl._smpl import SMPL, Mesh
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
import models.data.config as cfg

from models.utils.renderer import Renderer, visualize_reconstruction_no_text, visualize_reconstruction
from metro.utils.geometric_layers import orthographic_projection
from models.utils.logger import setup_logger
#from models.utils.miscellaneous import set_seed

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(200),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(200),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


def run(args, image_list, _gaze_network, bodyrenderer_network, renderer, smpl, mesh_sampler):

    _gaze_network.eval()
    smpl.eval()

    for image_file in image_list:
        image = Image.open(image_file)
        img_tensor = transform(image)
        img_visual = transform_visualize(image)

        batch_imgs = torch.unsqueeze(img_tensor, 0).cuda(args.device)
        batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda(args.device)

        # forward-pass
        direction = _gaze_network(batch_imgs, smpl, mesh_sampler, render=True)
        pred_camera, pred_3d_joints, _, _, pred_vertices, _, _, _ = bodyrenderer_network(batch_imgs, smpl, mesh_sampler)

        visual_imgs = visualize_mesh_no_text( renderer, 
                                              batch_visual_imgs[0],
                                              pred_vertices[0].detach(), 
                                              pred_camera.detach(),
                                              )

        # obtain 3d joints from full mesh
        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

        pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

        visual_imgs = visual_imgs.transpose(1,2,0)
        visual_imgs = np.asarray(visual_imgs)

        # visualize gaze direction
        direction = direction.cpu()
        print(direction)
        gazedir_2d = direction[0, 0:2].detach().numpy()
        print(gazedir_2d)

        # convert by projection : 3D joint to 2D joint
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

        pred_head = pred_2d_joints[:, 13,:]
        pred_head =((pred_head + 1) * 0.5) * 224
        print(pred_head)

        #gazedir_2d /= np.linalg.norm(gazedir_2d)
        head_center_x = pred_head[0][0]
        head_center_y = pred_head[0][1]

        # 2D head position
        head_center = (int(head_center_x), int(head_center_y))
        des = (head_center[0] + int(gazedir_2d[0]*20), int(head_center[1] + gazedir_2d[1]*20))
        visual_imgs = visualize_mesh(renderer, batch_visual_imgs[0], pred_camera.detach(), pred_2d_joints[0])
        visual_imgs = visual_imgs.transpose(1,2,0)
        visual_imgs = np.asarray(visual_imgs)

        # draw allow line from head to gaze direction
        visual_imgs = cv2.arrowedLine(visual_imgs.copy(), head_center, des, (0, 255, 0), 3, tipLength=0.3)
        temp_fname = image_file[:-4] + '_pred.jpg'
        print("save to ", temp_fname)
        cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

    return

def visualize_mesh( renderer,
                    images,
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    #batch_size = pred_keypoints_2d.shape[0]
    # Do visualization for the first 6 images of the batch
    img = images.cpu().numpy().transpose(1,2,0)
    # Get LSP keypoints from the full list of keypoints
    pred_keypoints_2d_ = pred_keypoints_2d.cpu().detach().numpy()
    # Get predict vertices for the particular example
    cam = pred_camera.cpu().numpy()
    # Visualize reconstruction and detected pose
    rend_img = visualize_reconstruction(img, 224, pred_keypoints_2d_, cam, renderer)
    rend_img = rend_img.transpose(2,0,1)
    #rend_imgs.append(torch.from_numpy(rend_img))   

    return rend_img




def visualize_mesh_no_text( renderer,
                    images,
                    pred_vertices, 
                    pred_camera):
    """Tensorboard logging."""
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    # Visualize reconstruction only
    rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img



def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./', type=str, 
                        help="image data")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--model_checkpoint", default='output/checkpoint2212/checkpoint-6-54572/state_dict.bin', type=str, required=False,
                        help="Path to wholebodygaze checkpoint for inference.")
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
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")

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
    renderer = Renderer(faces=mesh_smpl.faces.cpu().numpy())

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
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            # model_class = METRO
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # for ここまで
        # init ImageNet pre-trained backbone model
        # arch default='hrnet-w64'
        if args.arch=='hrnet':
            hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
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

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        # from metro.modeling.bert import METRO_Body_Network as METRO_Network
        # ここでモデルの初期化
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
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



    bodyrenderer_network = copy.deepcopy(_metro_network)



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

    # loading images
    image_list = []

    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_list.append(args.image_file_or_path+'/'+filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))



    logger.info("Run")

    run(args, image_list, _gaze_network, bodyrenderer_network, renderer, mesh_smpl, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
