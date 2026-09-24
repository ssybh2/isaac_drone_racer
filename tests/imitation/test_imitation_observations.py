import math

import torch

from tasks.drone_racer.mdp.imitation_observations import _rotvec_to_quat_wxyz


def test_rotvec_to_quaternion_identity_and_unit_norm():
    q0 = _rotvec_to_quat_wxyz(torch.zeros(2, 3))
    torch.testing.assert_close(
        q0,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
        atol=1.0e-6,
        rtol=0.0,
    )

    rotvec = torch.tensor([[0.0, 0.0, math.radians(10.0)]])
    q = _rotvec_to_quat_wxyz(rotvec)
    torch.testing.assert_close(
        torch.linalg.vector_norm(q, dim=-1),
        torch.ones(1),
        atol=1.0e-6,
        rtol=0.0,
    )
