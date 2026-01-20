"""
Code adapted from https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions
See Equations 85-90 in https://arxiv.org/pdf/1710.03746.pdf for more details.
"""

import torch
import torch.nn as nn

# Bessel function polynomial approximation coefficients from https://omlc.org/software/mc/conv-src/convnr.c
bessel0_exp_scaled_coeffs_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
bessel0_exp_scaled_coeffs_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]


def horners_method(coeffs, x):
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(coeffs[0])
    for i in range(1, len(coeffs)):
        z.mul_(x).add_(coeffs[i])
    return z

def bessel0_exp_scaled(x, eps=1e-8):
    abs_x = torch.abs(x).clamp(min=eps)

    z1 = (abs_x / 3.75) ** 2
    z2 = 3.75 / abs_x
    sqrt_x = torch.sqrt(abs_x)

    I_0_bar_a = horners_method(bessel0_exp_scaled_coeffs_a, z1) / torch.exp(abs_x)
    I_0_bar_b = horners_method(bessel0_exp_scaled_coeffs_b, z2) / sqrt_x

    mask = abs_x <= 3.75
    I_0_bar = torch.where(mask, I_0_bar_a, I_0_bar_b)

    return I_0_bar


def torch_trapezoid_integral(func,
                             func_args,
                             from_x,
                             to_x,
                             num_traps):
    with torch.no_grad():
        range = torch.arange(num_traps, dtype=func_args.dtype, device=func_args.device)
        x = (range * ((to_x-from_x) / (num_traps - 1)) + from_x).view(1, num_traps)  # (x values: [from_x, to_x])
        weights = torch.empty((1, num_traps), dtype=func_args.dtype, device=func_args.device).fill_(1)
        weights[0, 0] = 1/2
        weights[0, -1] = 1/2
        y = func(x, func_args)

        return torch.sum(y * weights, dim=1) * (to_x - from_x)/(num_traps - 1)

def integrand_normconst_forward_exp_scaled(u, s):
    # s is sorted from big to small
    factor1 = (s[:, [1]] - s[:, [2]]) * 0.5 * (1 - u)
    factor1 = bessel0_exp_scaled(factor1)

    factor2 = (s[:, [1]] + s[:, [2]]) * 0.5 * (1 + u)
    factor2 = bessel0_exp_scaled(factor2)

    factor3_input = (s[:, [2]] + s[:, [0]]) * (u - 1)

    factor3 = torch.exp(factor3_input)

    integrand_c_bar = factor1 * factor2 * factor3
    return integrand_c_bar


def integrand_dlognormconst_ds_backward(u, s):
    s_i = torch.max(s[:, 1:], dim=1, keepdim=True).values
    s_j = torch.min(s[:, 1:], dim=1, keepdim=True).values
    s_k = s[:, [0]]

    factor1 = (s_i - s_j) * 0.5 * (1 - u)
    factor1 = bessel0_exp_scaled(factor1)

    factor2 = (s_i + s_j) * 0.5 * (1 + u)
    factor2 = bessel0_exp_scaled(factor2)

    factor3 = torch.exp((s_j + s_k) * (u - 1))

    integrand_dlogc_dcs_k = factor1 * factor2 * factor3 * u  # Don't have u-1 here because this is integrand of dc_bar(S)/ds_k + c_bar(S).
    return integrand_dlogc_dcs_k


class LogMFNormConstant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, S):
        num_traps = 512  # Number of trapezoids + 1 for integral

        c_bar = 0.5 * torch_trapezoid_integral(func=integrand_normconst_forward_exp_scaled,
                                               func_args=S,
                                               from_x=-1,
                                               to_x=1,
                                               num_traps=num_traps)  # c_bar(S), shape is (B,)
        ctx.save_for_backward(S, c_bar)  # Save for gradient computation in backward pass

        c_bar = c_bar.clamp(min=1e-12)  # Prevent log(0)
        log_c_bar = torch.log(c_bar)  # log(c_bar(S))
        log_trace_S = torch.sum(S, dim=1)  # tr(S)

        log_c = log_c_bar + log_trace_S  # log(c(S)) = log(c_bar(S)) + tr(S)
        return log_c

    @staticmethod
    def backward(ctx, grad_log_c):
        S, c_bar = ctx.saved_tensors  # S is proper singular values, c_bar is exp scaled log norm constant.
        num_traps = 512  # Number of trapezoids + 1 for integral

        dc_bar_dS = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            S_shifted = torch.cat((S[:, i:], S[:, :i]), dim=1)  # Cyclic shifts of singular values
            dc_bar_dS[:, i] = 0.5 * torch_trapezoid_integral(func=integrand_dlognormconst_ds_backward,
                                                             func_args=S_shifted,
                                                             from_x=-1,
                                                             to_x=1,
                                                             num_traps=num_traps)  # dc_bar(S) / ds_k + c_bar(S)
        dlogc_dS = dc_bar_dS / c_bar.view(-1, 1).clamp(min=1e-12)  # dlog(c(S)) / ds_k
        grad_S = dlogc_dS * grad_log_c.view(-1, 1)
        return grad_S.view(-1, 3)


def debug_check_tensors(**tensors):
    for name, t in tensors.items():
        if not torch.isfinite(t).all():
            print(f"Non-finite in {name}: nan={torch.isnan(t).any().item()}, inf={torch.isinf(t).any().item()}, min={t.min().item()}, max={t.max().item()}")



def matrix_fisher_nll(pred_F,
                      R_mode,   # 単一の直交行列 (U V^T)
                      S_diag,   # (N, 3) 対角成分
                      R_target,
                      overreg=1.025):

    debug_check_tensors(S_diag=S_diag, pred_F=pred_F, R_mode=R_mode, R_target=R_target)



    # flatten batch and frames
    N = R_mode.shape[0] * R_mode.shape[1]
    pred_F = pred_F.view(N, 3, 3)
    R_mode = R_mode.view(N, 3, 3)
    S_diag = S_diag.view(N, 3)
    R_target = R_target.view(N, 3, 3)

    # Proper singular values を計算
    with torch.no_grad():
        s3sign = torch.det(R_mode.detach().cpu()).to(S_diag.device)  # det(UV^T) = ±1
    pred_S_proper = S_diag.clone()
    pred_S_proper[..., 2] *= s3sign

    # 正規化定数
    log_norm_constant = LogMFNormConstant.apply(pred_S_proper)

    # 尤度項
    log_exponent = -torch.matmul(pred_F.view(-1, 1, 9), R_target.view(-1, 9, 1)).view(-1)

    #return log_exponent + overreg * log_norm_constant
    loss = log_exponent + overreg * log_norm_constant
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if torch.isnan(loss).any():
        print("Warning: NaN detected in loss and replaced with 0.")

    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    if torch.isinf(loss).any():
        print("Warning: Inf detected in loss and replaced with 0.")

    return loss



class SO3GeodesicLoss(nn.Module):
    """
    Geodesic loss on SO(3):
        d(R1, R2) = acos((tr(R1^T R2) - 1) / 2)

    - fully differentiable
    - numerically stable
    - no SVD
    """

    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, R_pred, R_gt):
        """
        Args:
            R_pred: (..., 3, 3)
            R_gt:   (..., 3, 3)

        Returns:
            loss: scalar or (...) depending on reduction
        """
        # relative rotation
        R_rel = R_pred.transpose(-1, -2) @ R_gt

        # trace
        trace = (
            R_rel[..., 0, 0]
            + R_rel[..., 1, 1]
            + R_rel[..., 2, 2]
        )

        # cos(theta)
        cos_theta = (trace - 1.0) * 0.5
        cos_theta = torch.clamp(
            cos_theta,
            -1.0 + self.eps,
            1.0 - self.eps
        )

        theta = torch.acos(cos_theta)

        if self.reduction == "mean":
            return theta.mean()
        elif self.reduction == "sum":
            return theta.sum()
        elif self.reduction == "none":
            return theta
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
