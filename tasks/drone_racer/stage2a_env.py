"""Stage2A environment integration point.

This file intentionally keeps training logic untouched.
It will replace oracle target_pos_b with perception estimated target pose.
"""


def stage2a_target_observation(env):
    """Future observation hook.

    Returns:
        estimated target position in body frame from PerfectGateCornerSensor + PnP.
    """
    raise NotImplementedError(
        "Connect GatePoseEstimator after validating camera extrinsics."
    )
