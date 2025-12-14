import torch

def svd_decompose_rotations(R: torch.Tensor, device=None):
    """
    R: (B, T, 3, 3)
    Proper SVD decomposition with det correction.
    """
    if device is None:
        device = R.device

    B, T, _, _ = R.shape

    #R, _ = torch.linalg.qr(R)

    # --- バッチSVD ---
    U, S, Vh = torch.linalg.svd(R)  # (B,T,3,3), (B,T,3), (B,T,3,3)

    # det補正
    detU = torch.sign(torch.det(U))                       # (B,T)
    detV = torch.sign(torch.det(Vh.transpose(-1, -2)))     # (B,T)

    # Proper SVD の補正
    U_proper = U.clone()
    S_proper = S.clone()
    Vh_proper = Vh.clone()

    # U の最後の列を detU で補正
    U_proper[..., :, 2] *= detU.unsqueeze(-1)

    # Vh の最後の行を detV で補正
    Vh_proper[..., 2, :] *= detV.unsqueeze(-1)

    # S の最後の成分を detU*detV で補正
    S_proper[..., 2] *= detU * detV
    S_proper = torch.clamp(S_proper, min=1e-6, max=10)


    # Proper 回転モード行列
    R_mode = U_proper @ Vh_proper

    F_proper = U_proper @ torch.diag_embed(S_proper) @ Vh_proper

    return F_proper, U_proper, S_proper, Vh_proper, R_mode






