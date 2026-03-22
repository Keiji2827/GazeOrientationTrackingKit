"""
Code adapted from https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions
See Equations 85-90 in https://arxiv.org/pdf/1710.03746.pdf for more details.
"""

import sys
import torch
import torch.nn as nn

# Bessel function polynomial approximation coefficients from https://omlc.org/software/mc/conv-src/convnr.c
bessel0_exp_scaled_coeffs_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
bessel0_exp_scaled_coeffs_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]


def horners_method(coeffs, x):
    z = torch.full_like(x, float(coeffs[0]))
    for i in range(1, len(coeffs)):
        z = z * x + coeffs[i]
    return z

def bessel0_exp_scaled(x, eps=1e-12):
    """
    exp(-|x|) * I0(|x|) の近似。
    元コードと同じ近似式だが、torch.where で両枝を同時評価せず、
    mask ごとに計算して不要枝の数値発散を避ける。
    """
    abs_x = torch.abs(x).clamp(min=eps)
    out = torch.empty_like(abs_x)

    mask_small = abs_x <= 3.75
    mask_large = ~mask_small

    if mask_small.any():
        xs = abs_x[mask_small]
        z1 = (xs / 3.75) ** 2
        poly_a = horners_method(bessel0_exp_scaled_coeffs_a, z1)
        out[mask_small] = poly_a / torch.exp(xs)

    if mask_large.any():
        xl = abs_x[mask_large]
        z2 = 3.75 / xl
        sqrt_x = torch.sqrt(xl)
        poly_b = horners_method(bessel0_exp_scaled_coeffs_b, z2)
        out[mask_large] = poly_b / sqrt_x

    return out



def torch_trapezoid_integral(func, func_args, from_x, to_x, num_traps):
    """
    積分は grad 不要なので no_grad のまま。
    ただし内部計算は float64 に上げて数値安定性を改善する。
    """
    orig_dtype = func_args.dtype
    calc_dtype = torch.float64 if orig_dtype in (torch.float16, torch.float32, torch.float64) else orig_dtype

    with torch.no_grad():
        func_args_ = func_args.to(calc_dtype)

        grid = torch.arange(num_traps, dtype=calc_dtype, device=func_args.device)
        x = (grid * ((to_x - from_x) / (num_traps - 1)) + from_x).view(1, num_traps)

        weights = torch.ones((1, num_traps), dtype=calc_dtype, device=func_args.device)
        weights[0, 0] = 0.5
        weights[0, -1] = 0.5

        y = func(x, func_args_)

        out = torch.sum(y * weights, dim=1) * ((to_x - from_x) / (num_traps - 1))
        return out.to(orig_dtype)

def integrand_normconst_forward_exp_scaled(u, s):
    # s is sorted from big to small
    factor1 = (s[:, [1]] - s[:, [2]]) * 0.5 * (1 - u)
    factor1 = bessel0_exp_scaled(factor1)

    factor2 = (s[:, [1]] + s[:, [2]]) * 0.5 * (1 + u)
    factor2 = bessel0_exp_scaled(factor2)

    factor3_input = (s[:, [2]] + s[:, [0]]) * (u - 1)

    # 本来この項は多くの場合 0 以下だが、数値揺れで極端に大きい正値になった場合だけ抑制
    factor3_input = torch.clamp(factor3_input, min=-80.0, max=80.0)
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

    factor3_input = (s_j + s_k) * (u - 1)
    factor3_input = torch.clamp(factor3_input, min=-80.0, max=80.0)
    factor3 = torch.exp(factor3_input)

    integrand_dlogc_dcs_k = factor1 * factor2 * factor3 * u
    return integrand_dlogc_dcs_k


class LogMFNormConstant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, S):
        """
        正規化定数計算は double で行い、返り値だけ元 dtype に戻す。
        """
        input_dtype = S.dtype
        S64 = S.to(torch.float64)

        num_traps = 512

        c_bar64 = 0.5 * torch_trapezoid_integral(
            func=integrand_normconst_forward_exp_scaled,
            func_args=S64,
            from_x=-1,
            to_x=1,
            num_traps=num_traps
        ).to(torch.float64)

        # 積分誤差で負や 0 に触れるのを防ぐ
        c_bar64 = c_bar64.clamp(min=1e-300)

        ctx.save_for_backward(S64, c_bar64)
        ctx.input_dtype = input_dtype

        log_c_bar64 = torch.log(c_bar64)
        log_trace_S64 = torch.sum(S64, dim=1)
        log_c64 = log_c_bar64 + log_trace_S64

        return log_c64.to(input_dtype)

    @staticmethod
    def backward(ctx, grad_log_c):
        S64, c_bar64 = ctx.saved_tensors
        grad_log_c64 = grad_log_c.to(torch.float64)

        num_traps = 512

        dc_bar_dS64 = torch.empty((S64.shape[0], 3), dtype=torch.float64, device=S64.device)
        for i in range(3):
            S_shifted64 = torch.cat((S64[:, i:], S64[:, :i]), dim=1)
            dc_bar_dS64[:, i] = 0.5 * torch_trapezoid_integral(
                func=integrand_dlognormconst_ds_backward,
                func_args=S_shifted64,
                from_x=-1,
                to_x=1,
                num_traps=num_traps
            ).to(torch.float64)

        denom = c_bar64.view(-1, 1).clamp(min=1e-300)
        dlogc_dS64 = dc_bar_dS64 / denom
        grad_S64 = dlogc_dS64 * grad_log_c64.view(-1, 1)

        return grad_S64.to(ctx.input_dtype)




def matrix_fisher_nll(
    pred_F,
    R_mode,   # 単一の直交行列 (U V^T)
    S_diag,   # (..., 3)
    R_target,
    overreg=1.025
):
    # flatten batch and frames
    N = R_mode.shape[0] * R_mode.shape[1]
    pred_F = pred_F.reshape(N, 3, 3)
    R_mode = R_mode.reshape(N, 3, 3)
    S_diag = S_diag.reshape(N, 3)
    R_target = R_target.reshape(N, 3, 3)

    # Proper singular values を計算
    # det(UV^T) は理論上 ±1 なので、符号だけを使って安定化
    with torch.no_grad():
        det_R = torch.linalg.det(R_mode.detach().to(torch.float64)).to(S_diag.device)
        s3sign = torch.where(det_R >= 0, torch.ones_like(det_R), -torch.ones_like(det_R)).to(S_diag.dtype)

    pred_S_proper = S_diag.clone()
    pred_S_proper[..., 2] = pred_S_proper[..., 2] * s3sign

    # 正規化定数
    log_norm_constant = LogMFNormConstant.apply(pred_S_proper)

    # 尤度項
    log_exponent = -torch.matmul(
        pred_F.reshape(-1, 1, 9),
        R_target.reshape(-1, 9, 1)
    ).reshape(-1)

    loss = log_exponent + overreg * log_norm_constant
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
