import torch


def project_points_pinhole(points_c: torch.Tensor, K: torch.Tensor):
    """Project camera-frame 3D points to pixels.

    points_c: (...,3)
    K: (3,3)
    return: (...,2)
    """
    z = points_c[..., 2].clamp(min=1e-6)
    x = points_c[..., 0] / z
    y = points_c[..., 1] / z

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x + cx
    v = fy * y + cy
    return torch.stack([u, v], dim=-1)


def transform_points(points, R, t):
    """World/Gate to camera transform."""
    return (R @ points.T).T + t
