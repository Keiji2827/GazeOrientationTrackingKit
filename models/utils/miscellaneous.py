import os
import torch
import random
#import cv2
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.dataloader.gafa_loader import create_gafa_dataset



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




def load_from_state_dict(args, smpl, mesh_sampler):

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

    _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

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
    _metro_network.to(args.device)

    return _metro_network



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
        'living_room/004',
        'kitchen/1015_4',
        'kitchen/1022_4',
        'library/1028_2',
        'library/1028_5',
        'library/1026_3',
        'courtyard/004',
        'courtyard/005',
        'lab/1013_1',
        'lab/1014_1',
                    ]
        random.shuffle(exp_names)
        # GAFA dataset
        dset = create_gafa_dataset(exp_names=exp_names, n_frames=args.n_frames)

    return dset



# Not used currently(25/12/14)
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
