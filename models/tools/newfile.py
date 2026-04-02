import argparse
import sys
import datetime
import time
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
from models.smpl._smpl import SMPL, Mesh
from models.bert.modeling_gabert import GAZEFROMBODY
from models.utils.geometric_layers import rotation_from_two_vectors
from models.utils.matrix_fisher_loss import SO3GeodesicLoss, matrix_fisher_nll
from models.utils.Angle_Error_loss import CosLoss, CosLossSingle
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import save_checkpoint, load_from_state_dict, create_dataset, create_testdataset, create_valdataset


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
    parser.add_argument("--model_checkpoint", default='', 
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
    parser.add_argument("--num_train_epochs", default=8, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--lr', "--learning_rate", default=1e-5, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_init_epoch", default=0, type=int, 
                        help="initial epoch number.")
    parser.add_argument("--train_batch_size", default=4, type=int, 
                        help="Batch size for training.")
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
    parser.add_argument('--no_use_lstm', action='store_true', default=False,
                        help="ablation study: without LSTM, just use cumulative rotations")
    parser.add_argument('--no_use_MF', action='store_true', default=False,
                        help="ablation study: without matrix fisher loss")

    args = parser.parse_args()
    return args

def main(args):

    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    if args.no_use_lstm:
        print("Ablation study: Using cumulative rotations without LSTM")
    if args.no_use_MF:
        print("Ablation study: Not using matrix fisher loss")

    train_batch_size = args.train_batch_size

    print("Command Lines: ", ' '.join(sys.argv))

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

    if not args.model_checkpoint == '':
        state_dict = torch.load(args.model_checkpoint)
        _gaze_network.load_state_dict(state_dict)
        del state_dict

    if not args.test:
        print("Train mode")
        dset = create_dataset(args)
        train_idx, val_idx = np.arange(0, int(len(dset)*0.99)), np.arange(int(len(dset)*0.99), len(dset))
        train_dset, val_dset = random_split(dset, [len(train_idx), len(val_idx)])

        train_dataloader = DataLoader(
            train_dset, batch_size=train_batch_size, num_workers=2, pin_memory=True, persistent_workers=True, shuffle=True
        )
        #val_dataloader = DataLoader(
        #    val_dset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
        #)
        dset = create_valdataset(args)
        test_dataloader = DataLoader(
            dset, batch_size=4, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True
        )
        # Training
        train(args, train_dataloader, test_dataloader, _gaze_network, smpl, mesh_sampler)



    else: 
        print("Test mode")
        print("Load checkpoint from {}".format(args.model_checkpoint))
        dset = create_testdataset(args)
        test_dataloader = DataLoader(
            dset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True
        )

        val = validate(args, test_dataloader, _gaze_network, smpl, mesh_sampler)

        print("val:", torch.rad2deg(torch.tensor(val)))


    return 0


def train(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler):
    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)


    frame = args.n_frames // 2
    epochs = args.num_train_epochs

    # optimizer settings
    backbone_params = list(_gaze_network.BertLayer.bert.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    bertlayer_other_params = [p for p in _gaze_network.BertLayer.parameters() if id(p) not in backbone_param_ids]

    optimizer = torch.optim.AdamW(
        [
        {"params": backbone_params, "lr": args.lr * 0.02},
        {"params": bertlayer_other_params, "lr": args.lr},
        {"params": _gaze_network.HeadMFLayer.parameters(), "lr": args.lr},
        {"params": _gaze_network.LSTMlayer.parameters(), "lr": args.lr}
    ],
        betas=(0.9, 0.999), weight_decay=0
    )
    # 
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = param_group["lr"]

    # ===== scheduler settings =====
    total_steps = len(train_dataloader) * args.num_train_epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup（必要なら調整）
    # ===== get learning rate scale =====
    def get_lr_scale(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))


    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    criterion_seq = CosLoss().cuda(args.device)
    criterion_dir = CosLossSingle().cuda(args.device)
    criterion_mdir = CosLossSingle().cuda(args.device)
    criterion_Rot  = SO3GeodesicLoss().cuda(args.device)

    for epoch in range(args.num_init_epoch, epochs):
        log_losses = AverageMeter()
        log_seq = AverageMeter()
        log_dir = AverageMeter()
        log_mdir = AverageMeter()
        log_Rot = AverageMeter()
        log_MF = AverageMeter()
        for iteration, batch in enumerate(train_dataloader):

            log_con = AverageMeter()

            iteration += 1
            _gaze_network.train()

            batch_imgs = batch['image'].cuda(args.device, non_blocking=True)
            gaze_dir = batch['gaze_dir'].cuda(args.device, non_blocking=True)
            head_dir = batch["head_dir"].cuda(args.device, non_blocking=True)
            #head_mask = batch["head_mask"].cuda(args.device)

            batch_size = batch_imgs.size(0)

            #for param_group in optimizer.param_groups:
            #    param_group["lr"] = args.lr

            data_time.update(time.time() - end)

            # forward-pass
            directions, mdir, R_mode, S_diag, pred_F, d_corr = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=True)

            if not args.no_use_MF:
                confidence = S_diag.sum(dim=-1).detach()
                confidence = confidence / (confidence.max() + 1e-8)

                # 数値安定化（STE）
                S_safe = torch.clamp(S_diag, min=0.0, max=8.0)
                S_diag = S_safe + (S_diag - S_diag.detach())

            # compute target rotation matrices from gaze directions
            R_target = rotation_from_two_vectors(gaze_dir)

            # loss
            loss_seq = criterion_seq(directions, gaze_dir).mean()
            loss_dir = criterion_dir(directions[:,frame,:],gaze_dir[:,frame,:]).mean()
            loss_mdir = criterion_mdir(mdir, head_dir[:, frame,:]).mean()
            loss_Rot = criterion_Rot(R_mode, R_target).mean()
            if not args.no_use_MF:
                loss_MF = matrix_fisher_nll(pred_F, R_mode, S_diag, R_target).mean()
                loss_MF = torch.clamp(loss_MF, -50.0, 50.0)
            else:
                loss_MF = torch.tensor(0.0).cuda(args.device)
                confidence = torch.tensor(0.0).cuda(args.device)

            a = 4.
            b = 5.
            m = 2.
            c = 1.
            d = 0.01

            # 学習初期はMF lossの重みを小さくして、徐々に増やす（安定化のため）
            if epoch == 0:
                mf_weight = 0.0
            elif epoch == 1:
                mf_weight = 0.05
            elif epoch == 2:
                mf_weight = 0.1
            elif epoch == 3:
                mf_weight = 0.25
            elif epoch == 4:
                mf_weight = 0.5
            else:
                mf_weight = 1.0


            loss = ((a)*loss_seq  + b*loss_dir + m*loss_mdir + c*loss_Rot + mf_weight*d*loss_MF)
            #loss = confidence*((a)*loss_seq  + b*loss_dir + c*loss_Rot + d*loss_MF)
            loss = loss.mean()
            # update logs
            log_losses.update(loss.item(), batch_size)
            log_seq.update(torch.rad2deg(loss_seq).item(), batch_size)
            log_dir.update(torch.rad2deg(loss_dir).item(), batch_size)
            log_mdir.update(torch.rad2deg(loss_mdir).item(), batch_size)
            log_Rot.update(torch.rad2deg(loss_Rot).item(), batch_size)
            log_MF.update(loss_MF.item(), batch_size)
            log_con.update(confidence.mean().item(), batch_size)

            # back prop
            optimizer.zero_grad(set_to_none=True)

            # ===== 軽量 NaN チェック（loss内訳の自動特定）=====
            if not torch.isfinite(loss):
                bad_terms = []

                if not torch.isfinite(loss_seq):
                    bad_terms.append(f"loss_seq={loss_seq.item()}")
                if not torch.isfinite(loss_dir):
                    bad_terms.append(f"loss_dir={loss_dir.item()}")
                if not torch.isfinite(loss_mdir):
                    bad_terms.append(f"loss_mdir={loss_mdir.item()}")
                if not torch.isfinite(loss_Rot):
                    bad_terms.append(f"loss_Rot={loss_Rot.item()}")
                if not torch.isfinite(loss_MF):
                    bad_terms.append(f"loss_MF={loss_MF.item()}")

                # 個別lossは有限なのに total loss だけ非有限な場合も拾う
                if len(bad_terms) == 0:
                    bad_terms.append(f"total_loss={loss.item()} (components look finite)")

                print(
                    f"[WARN] Non-finite loss detected. "
                    f"epoch={epoch}, iter={iteration}, "
                    f"bad_terms: {', '.join(bad_terms)}"
                )
                continue

            loss.backward()

            # ===== 軽量 NaN チェック（loss内訳の自動特定）=====
            if not torch.isfinite(loss):
                bad_terms = []

                if not torch.isfinite(loss_seq):
                    bad_terms.append(f"loss_seq={loss_seq.item()}")
                if not torch.isfinite(loss_dir):
                    bad_terms.append(f"loss_dir={loss_dir.item()}")
                if not torch.isfinite(loss_mdir):
                    bad_terms.append(f"loss_mdir={loss_mdir.item()}")
                if not torch.isfinite(loss_Rot):
                    bad_terms.append(f"loss_Rot={loss_Rot.item()}")
                if not torch.isfinite(loss_MF):
                    bad_terms.append(f"loss_MF={loss_MF.item()}")

                # 個別lossは有限なのに total loss だけ非有限な場合も拾う
                if len(bad_terms) == 0:
                    bad_terms.append(f"total_loss={loss.item()} (components look finite)")

                print(
                    f"[WARN] Non-finite loss detected. "
                    f"epoch={epoch}, iter={iteration}, "
                    f"bad_terms: {', '.join(bad_terms)}"
                )
                continue


            if _gaze_network.BertLayer.bert.backbone.conv1.weight.grad is not None:
                _gaze_network.BertLayer.bert.backbone.conv1.weight.grad.data.clamp_(-1.0, 1.0)
            torch.nn.utils.clip_grad_norm_(_gaze_network.parameters(), max_norm=0.5)

            # ===== update learning rate (cosine + warmup) =====
            # summary: 学習率をウォームアップとコサイン減衰でスケジュールする
            global_step = epoch * max_iter + (iteration - 1)
            lr_scale = get_lr_scale(global_step)

            # 学習率をスケールして更新
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_scale

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if iteration%args.logging_steps == 0 or iteration == max_iter:
                now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(
                    f"date: {now},"
                    f" eta: {eta_string}, epoch: {epoch}, iter: {iteration},"
                    f" loss: {log_losses.avg:.4f}, angle: {(log_dir.avg+log_seq.avg)/2:.3f},"
                    f" Rot: {log_Rot.avg:.3f}, MF: {log_MF.avg:.3f},"
                    f" con: {log_con.avg:.6f}",
                    f" lr: {optimizer.param_groups[1]['lr']:.1e}".replace("e-0", "e-")
                    )

        checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
        print("save trained model at ", checkpoint_dir)
        val = validate(args, val_dataloader, 
                            _gaze_network, 
                            smpl, 
                            mesh_sampler,
                            in_train=True
                )
        print("val:", torch.rad2deg(torch.tensor(val)))


    return 0

def validate(args, val_dataloader, gaze_network, smpl, mesh_sampler, in_train=False):
    max_iter = len(val_dataloader)
    end = time.time()
    batch_time = AverageMeter()

    log_losses = AverageMeter()
    log_losses_front = AverageMeter()
    log_losses_back = AverageMeter()

    gaze_network.eval()
    frame = args.n_frames // 2
    criterion = CosLossSingle().cuda(args.device)

    # カウント用
    count_front = 0
    count_back = 0

    print("len of dataset:", max_iter)
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
            direction, S_diag = gaze_network(batch_imgs, smpl, mesh_sampler, is_train=False)

            confidence = torch.tensor(0.0).cuda(args.device)
            if not args.no_use_MF:
                confidence = S_diag.sum(dim=-1).detach()
                confidence = confidence / (confidence.max() + 1e-8)

            loss = criterion(direction,gaze_dir).mean()

            # update logs
            log_losses.update(loss.item(), batch_size)
            # -------------------------
            # 方向判定（frontal / back）
            # -------------------------
            z = gaze_dir[:, 2]  # [B]

            mask_front = z > 0
            mask_back = z <= 0

            count_front += mask_front.sum().item()
            count_back += mask_back.sum().item()

            # frontal loss
            if mask_front.any():
                loss_front = criterion(direction[mask_front], gaze_dir[mask_front]).mean()
                log_losses_front.update(loss_front.item(), mask_front.sum().item())

            # back loss
            if mask_back.any():
                loss_back = criterion(direction[mask_back], gaze_dir[mask_back]).mean()
                log_losses_back.update(loss_back.item(), mask_back.sum().item())


            batch_time.update(time.time() - end)
            end = time.time()

            if iteration%args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(f"eta: {eta_string}, epoch: {epoch}, iter: {iteration}, "
                      f"loss: {log_losses.avg:.4f}," 
                      f"loss_front: {log_losses_front.avg:.3f},"
                      f"loss_back: {log_losses_back.avg:.3f},"
                      f"con: {confidence.mean().item():.3f}")

                if in_train:
                    return log_losses.avg
                    #return log_losses.avg
    print("val frontal:", torch.rad2deg(torch.tensor(log_losses_front.avg)))
    print("val back:", torch.rad2deg(torch.tensor(log_losses_back.avg)))
    print("count front:", count_front, " count back:", count_back)

    return log_losses.avg




if __name__ == "__main__":
    args = parse_args()
    main(args)
