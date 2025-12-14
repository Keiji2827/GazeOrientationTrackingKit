"""
Useful geometric operations, e.g. Orthographic projection and a differentiable Rodrigues formula

Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
import torch

def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    
    
def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


def rotation_matrices_from_gaze(gaze_dir):
    """
    gaze_dir: (batch_size, n_frames, 3)
    return: (batch_size, n_frames-1, 3, 3)
    """
    # 正規化
    gaze_dir = gaze_dir / gaze_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    v1 = gaze_dir[:, :-1, :]  # (batch, n_frames-1, 3)
    v2 = gaze_dir[:, 1:, :]   # (batch, n_frames-1, 3)

    # 内積と角度
    dot = (v1 * v2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)  # (batch, n_frames-1, 1)

    # 回転軸
    axis = torch.cross(v1, v2, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / axis_norm

    # 歪対称行列 K
    a_x, a_y, a_z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(a_x)
    K = torch.stack([
        torch.stack([zeros, -a_z, a_y], dim=-1),
        torch.stack([a_z, zeros, -a_x], dim=-1),
        torch.stack([-a_y, a_x, zeros], dim=-1)
    ], dim=-2)  # (batch, n_frames-1, 3, 3)

    I = torch.eye(3, device=gaze_dir.device).expand_as(K)

    # Rodriguesの公式
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]

    R = I + sin_t * K + (1 - cos_t) * (K @ K)

    return R  # (batch, n_frames-1, 3, 3)

def skew(v):
    """
    v: (..., 3)
    return: (..., 3, 3) skew-symmetric matrix
    """
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    return torch.stack([
        torch.stack([ zero, -vz,  vy], dim=-1),
        torch.stack([  vz, zero, -vx], dim=-1),
        torch.stack([-vy,  vx, zero], dim=-1)
    ], dim=-2)


def rotation_from_two_vectors(gaze_dir, eps=1e-6):

    # 正規化
    gaze_dir = gaze_dir / gaze_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    a = gaze_dir[:, :-1, :]  # (batch, n_frames-1, 3)
    b = gaze_dir[:, 1:, :]   # (batch, n_frames-1, 3)

    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)

    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(dim=-1, keepdim=True)         # (B, F-1, 1)
    c = c.unsqueeze(-1)   

    vx = skew(v)
    I = torch.eye(3, device=gaze_dir.device, dtype=gaze_dir.dtype)
    I = I.unsqueeze(0).unsqueeze(0)          # (1, 1, 3, 3)

    R = I + vx + (vx @ vx) * (1.0 / (1.0 + c + eps))

    return R


# --- 使用例 ---
batch_size, n_frames = 2, 5
gaze_dir = torch.randn(batch_size, n_frames, 3)
R = rotation_matrices_from_gaze(gaze_dir)
print(R.shape)  # (2, 4, 3, 3)









