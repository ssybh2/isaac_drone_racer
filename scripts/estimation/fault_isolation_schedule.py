"""Pure timing helpers for OpenVINS fault-isolation experiments."""

from __future__ import annotations


def resolve_motion_start_time_s(
    *,
    absolute_start_s: float,
    after_init_delay_s: float | None,
    initialized_time_s: float | None,
) -> float | None:
    """Resolve the simulation time at which commanded excitation should start.

    ``after_init_delay_s`` is intentionally optional so existing experiments keep
    their absolute-time behavior. When it is provided, motion remains disabled
    until an OpenVINS estimate has actually been observed, then starts after the
    requested delay. This keeps post-initialization propagation tests distinct
    from dynamic-initialization tests.
    """

    absolute_start_s = float(absolute_start_s)
    if absolute_start_s < 0.0:
        raise ValueError("absolute_start_s must be non-negative")

    if after_init_delay_s is None:
        return absolute_start_s

    after_init_delay_s = float(after_init_delay_s)
    if after_init_delay_s < 0.0:
        raise ValueError("after_init_delay_s must be non-negative")
    if initialized_time_s is None:
        return None

    return float(initialized_time_s) + after_init_delay_s
