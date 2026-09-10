"""PnP interface for Stage2A.

Initial implementation uses OpenCV IPPE. Later can be replaced by batched
Torch GPU planar PnP without changing the interface.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def solve_gate_pnp(object_points_g, image_points_uv, K):
    if cv2 is None:
        raise ImportError("opencv-python required for Stage2A PnP")

    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points_g, dtype=np.float32),
        np.asarray(image_points_uv, dtype=np.float32),
        np.asarray(K, dtype=np.float32),
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not ok:
        raise RuntimeError("PnP failed")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)
