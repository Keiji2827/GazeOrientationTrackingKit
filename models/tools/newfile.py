import argparse
from asyncio.log import logger
import sys
import random,datetime
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
    parser.add_argument("--num_train_epochs", default=10, type=int, 
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
        train_idx, val_idx = np.arange(0, int(len(dset)*0.95)), np.arange(int(len(dset)*0.95), len(dset))
        train_dset, val_dset = random_split(dset, [len(train_idx), len(val_idx)])

        train_dataloader = DataLoader(
            train_dset, batch_size=train_batch_size, num_workers=2, pin_memory=False, shuffle=True
        )
        #val_dataloader = DataLoader(
        #    val_dset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
        #)
        dset = create_valdataset(args)
        test_dataloader = DataLoader(
            dset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True
        )
        # Training
        train(args, train_dataloader, test_dataloader, _gaze_network, smpl, mesh_sampler)



    else: 
        print("Test mode")
        print("Load checkpoint from {}".format(args.model_checkpoint))
        dset = create_testdataset(args)
        test_dataloader = DataLoader(
            dset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True
        )

        val = validate(args, test_dataloader, _gaze_network, smpl, mesh_sampler)

        print("val:", torch.rad2deg(torch.tensor(val)))


    return 0



def train(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler):
    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)


    frame = args.n_frames // 2
    epochs = args.num_train_epochs
    optimizer = torch.optim.AdamW(
        #params=list(_gaze_network.parameters()),lr=args.lr, 
        [
        {"params": _gaze_network.BertLayer.parameters(), "lr": args.lr},
        {"params": _gaze_network.HeadMFLayer.parameters(), "lr": args.lr},
        {"params": _gaze_network.LSTMlayer.parameters(), "lr": args.lr}
    ],
        betas=(0.9, 0.999), weight_decay=0
    )

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    criterion_seq = CosLoss().cuda(args.device)
    criterion_dir = CosLossSingle().cuda(args.device)
    criterion_mdir = CosLossSingle().cuda(args.device)
    criterion_Rot  = SO3GeodesicLoss().cuda(args.device)

    nan_debug_done = False

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

            batch_imgs = batch['image'].cuda(args.device)
            gaze_dir = batch['gaze_dir'].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)
            #head_mask = batch["head_mask"].cuda(args.device)

            batch_size = batch_imgs.size(0)

            #for param_group in optimizer.param_groups:
            #    param_group["lr"] = args.lr

            data_time.update(time.time() - end)

            # forward-pass
            directions, mdir, R_mode, S_diag, pred_F, d_corr = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=True)

            def _check_tensor_simple(name, x):
                if not torch.isfinite(x).all():
                    return False
                return True

            def _check_tensor(name, x):
                if not torch.isfinite(x).all():
                    print(f"[DEBUG][{name}] has NaN/Inf")
                    print(f"  shape = {tuple(x.shape)}")
                    print(f"  nan count = {torch.isnan(x).sum().item()}")
                    print(f"  inf count = {torch.isinf(x).sum().item()}")
                    return False
                else:
                    with torch.no_grad():
                        print(
                            f"[DEBUG][{name}] "
                            f"min={x.min().item():.6e}, "
                            f"max={x.max().item():.6e}, "
                            f"mean={x.mean().item():.6e}"
                        )
                    return True

            ok1 = _check_tensor_simple("directions", directions)
            ok2 = _check_tensor_simple("mdir", mdir)
            ok3 = _check_tensor_simple("R_mode", R_mode)
            ok4 = _check_tensor_simple("S_diag(before clamp)", S_diag)
            ok5 = _check_tensor_simple("pred_F", pred_F)
            ok6 = _check_tensor_simple("d_corr", d_corr)

            if not (ok1 and ok2 and ok3 and ok4 and ok5 and ok6):
                if not nan_debug_done:
                    print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: network output already broken.")
                    _check_tensor("directions", directions)
                    _check_tensor("mdir", mdir)
                    _check_tensor("R_mode", R_mode)
                    _check_tensor("S_diag(before clamp)", S_diag)
                    _check_tensor("pred_F", pred_F)
                    _check_tensor("d_corr", d_corr)
                    nan_debug_done = True

                optimizer.zero_grad(set_to_none=True)
                continue

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

            # MF loss
            if not args.no_use_MF:
                try:
                    raw_loss_MF = matrix_fisher_nll(
                        pred_F, R_mode, S_diag, R_target
                    ).mean()
                except Exception as e:
                    if not nan_debug_done:
                        print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: matrix_fisher_nll failed: {e}")
                        _check_tensor("pred_F", pred_F)
                        _check_tensor("R_mode", R_mode)
                        _check_tensor("S_diag", S_diag)
                        _check_tensor("R_target", R_target)
                        nan_debug_done = True
                    raw_loss_MF = torch.tensor(float("nan"), device=batch_imgs.device, dtype=loss_seq.dtype)

                if torch.isfinite(raw_loss_MF):
                    loss_MF = torch.clamp(raw_loss_MF, min=-100.0, max=100.0)
                    mf_valid = True
                else:
                    if not nan_debug_done:
                        print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: raw_loss_MF is NaN/Inf.")
                        _check_tensor("pred_F", pred_F)
                        _check_tensor("R_mode", R_mode)
                        _check_tensor("S_diag", S_diag)
                        _check_tensor("R_target", R_target)
                        nan_debug_done = True

                    loss_MF = torch.zeros((), device=batch_imgs.device, dtype=loss_seq.dtype)
                    mf_valid = False
            else:
                loss_MF = torch.zeros((), device=batch_imgs.device, dtype=loss_seq.dtype)
                mf_valid = False
                confidence = torch.tensor(0.0, device=batch_imgs.device)

            a = 4.
            b = 5.
            m = 2.
            c = 1.
            d = 0.01

            # 学習初期はMF lossの重みを小さくして、徐々に増やす（安定化のため）
            if epoch == 0:
                mf_weight = 0.0
            elif epoch == 1:
                mf_weight = d * 0.05
            elif epoch == 2:
                mf_weight = d * 0.1
            elif epoch == 3:
                mf_weight = d * 0.25
            elif epoch == 4:
                mf_weight = d * 0.5
            else:
                mf_weight = d

            # nan/inf が出た step では MF を無効化
            if not mf_valid:
                mf_weight = 0.0

            loss = (
                a * loss_seq
                + b * loss_dir
                + m * loss_mdir
                + c * loss_Rot
                + mf_weight * loss_MF
            )
            loss = loss.mean()

            # ===== 追加4: 最終 loss の nan/inf 対策 =====
            if not torch.isfinite(loss):
                if not nan_debug_done:
                    print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: total loss is NaN/Inf.")
                    print(f"[DEBUG] loss_seq={loss_seq.item() if torch.isfinite(loss_seq) else loss_seq}")
                    print(f"[DEBUG] loss_dir={loss_dir.item() if torch.isfinite(loss_dir) else loss_dir}")
                    print(f"[DEBUG] loss_mdir={loss_mdir.item() if torch.isfinite(loss_mdir) else loss_mdir}")
                    print(f"[DEBUG] loss_Rot={loss_Rot.item() if torch.isfinite(loss_Rot) else loss_Rot}")
                    print(f"[DEBUG] loss_MF={loss_MF.item() if torch.isfinite(loss_MF) else loss_MF}")
                    print(f"[DEBUG] mf_weight={mf_weight}")
                    nan_debug_done = True

                optimizer.zero_grad(set_to_none=True)
                continue

            # logs
            log_losses.update(loss.item(), batch_size)
            log_seq.update(torch.rad2deg(loss_seq).item(), batch_size)
            log_dir.update(torch.rad2deg(loss_dir).item(), batch_size)
            log_mdir.update(torch.rad2deg(loss_mdir).item(), batch_size)
            log_Rot.update(torch.rad2deg(loss_Rot).item(), batch_size)
            log_MF.update(loss_MF.item(), batch_size)
            log_con.update(confidence.mean().item(), batch_size)

            # back prop
            optimizer.zero_grad()
            loss.backward() 

            bad_grad = False
            for name, p in _gaze_network.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    if not nan_debug_done:
                        print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: grad of {name} has NaN/Inf")
                        nan_debug_done = True
                    bad_grad = True
                    break

            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(_gaze_network.parameters(), max_norm=0.5)
            optimizer.step()

            bad_param = False
            for name, p in _gaze_network.named_parameters():
                if not torch.isfinite(p.data).all():
                    if not nan_debug_done:
                        print(f"[NaN DETECTED] epoch={epoch} iter={iteration}: parameter {name} became NaN/Inf after optimizer.step()")
                        nan_debug_done = True
                    bad_param = True
                    break

            if bad_param:
                break

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
                    f" con: {log_con.avg:.6f}")


                with torch.no_grad():
                    p99 = torch.quantile(d_corr.view(-1), 0.99).item()
                    p01 = torch.quantile(d_corr.view(-1), 0.01).item()

                #print(f"p01 = {p01:.6f}, p99 = {p99:.6f}")


            if False: #iteration%(args.logging_steps*10) == 0:
                val = validate(args, val_dataloader, 
                                    _gaze_network, 
                                    smpl, 
                                    mesh_sampler,
                                    in_train=True
                        )
                print("val:", torch.rad2deg(torch.tensor(val)))

                #checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
                print("save trained model at ", checkpoint_dir)


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

    #mse = AverageMeter()
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
