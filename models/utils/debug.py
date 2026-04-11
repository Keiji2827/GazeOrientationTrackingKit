import math
import torch

def tensor_stat_str(x, name):
    if x is None:
        return f"{name}: None"

    with torch.no_grad():
        finite = torch.isfinite(x)
        all_finite = finite.all().item()
        nan_any = torch.isnan(x).any().item()
        inf_any = torch.isinf(x).any().item()

        msg = (
            f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
            f"finite={all_finite}, nan={nan_any}, inf={inf_any}"
        )

        if x.numel() > 0:
            if all_finite:
                msg += (
                    f", min={x.min().item():.6e}"
                    f", max={x.max().item():.6e}"
                    f", mean={x.mean().item():.6e}"
                    f", absmax={x.abs().max().item():.6e}"
                )
            else:
                xf = x[finite]
                if xf.numel() > 0:
                    msg += (
                        f", finite_min={xf.min().item():.6e}"
                        f", finite_max={xf.max().item():.6e}"
                        f", finite_mean={xf.mean().item():.6e}"
                        f", finite_absmax={xf.abs().max().item():.6e}"
                    )
                else:
                    msg += ", all values are non-finite"

        return msg


def check_tensor_finite(name, x, epoch, iteration, stop_on_bad=False):
    if x is None:
        return False

    bad = not torch.isfinite(x).all()
    if bad:
        print(f"[TRACE][FORWARD] Non-finite tensor detected at epoch={epoch}, iter={iteration}")
        print(tensor_stat_str(x, name))
        if stop_on_bad:
            raise RuntimeError(f"Non-finite tensor detected: {name}")
    return bad


def check_scalar_finite(name, x, epoch, iteration, stop_on_bad=False):
    bad = not torch.isfinite(x)
    if bad:
        print(
            f"[TRACE][LOSS] Non-finite scalar detected at "
            f"epoch={epoch}, iter={iteration}, name={name}, value={x.item()}"
        )
        if stop_on_bad:
            raise RuntimeError(f"Non-finite scalar detected: {name}")
    return bad


def check_grads_finite(model, epoch, iteration, stop_on_bad=False):
    bad_found = False
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            print(
                f"[TRACE][BACKWARD] Non-finite gradient detected at "
                f"epoch={epoch}, iter={iteration}, param={name}"
            )
            print(tensor_stat_str(param.grad, f"{name}.grad"))
            bad_found = True
            if stop_on_bad:
                raise RuntimeError(f"Non-finite gradient detected: {name}")
            break
    return bad_found


def check_params_finite(model, epoch, iteration, stop_on_bad=False):
    bad_found = False
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(
                f"[TRACE][STEP] Non-finite parameter detected at "
                f"epoch={epoch}, iter={iteration}, param={name}"
            )
            print(tensor_stat_str(param.data, f"{name}.data"))
            bad_found = True
            if stop_on_bad:
                raise RuntimeError(f"Non-finite parameter detected: {name}")
            break
    return bad_found

# NaNやInfのトレースが有効かどうかを判定する関数
def trace_active(epoch, iteration):
    return (
        TRACE_NAN and
        epoch >= TRACE_START_EPOCH and
        (
            (epoch > TRACE_START_EPOCH) or
            (epoch == TRACE_START_EPOCH and iteration >= TRACE_START_ITER)
        )  and
        (iteration % TRACE_EVERY == 0)
    )



# ===== trace settings =====
TRACE_NAN = True          # False で通常運転
#TRACE_STOP = True         # True: 見つけた瞬間に停止
TRACE_EVERY = 1           # 負荷を下げるなら 10, 50, 100 など
TRACE_START_EPOCH = 0     # 特定epoch以降だけ有効化したいときに使う
TRACE_START_ITER = 1      # 特定iter以降だけ有効化したいときに使う

