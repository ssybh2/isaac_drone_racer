# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class Allocation:
    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant, F = k_f * omega^2
        - drag_coeff (float): Rotor drag-torque constant, tau_z = k_m * omega^2
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        if thrust_coeff <= 0.0:
            raise ValueError("thrust_coeff must be positive")

        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

        # ``compute`` first converts rotor speed to rotor thrust using
        #     F_i = k_f * omega_i^2.
        # Therefore the yaw row of the matrix must use k_m / k_f so that
        #     (k_m / k_f) * F_i = k_m * omega_i^2.
        # Using k_m directly here would multiply the two coefficients and
        # suppress yaw authority by a factor of k_f (~4.4e6 for the Swift
        # configuration), leaving the vehicle effectively uncontrolled in yaw.
        yaw_torque_per_thrust = drag_coeff / thrust_coeff

        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [
                    yaw_torque_per_thrust,
                    -yaw_torque_per_thrust,
                    yaw_torque_per_thrust,
                    -yaw_torque_per_thrust,
                ],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._inverse_allocation_matrix = torch.linalg.inv(A).unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque

    def allocate_wrench_rate_priority(self, wrench, omega_max):
        """Allocate [collective thrust, body moments] to bounded rotor speeds.

        The allocation mirrors the saturation behavior used by low-level racing
        controllers: body-moment authority is preserved before collective
        thrust whenever possible. The requested moment contribution is solved
        independently, scaled only if its rotor-thrust span is infeasible, then
        a common collective offset is chosen as close as possible to the
        requested collective thrust.

        Parameters
        ----------
        wrench:
            Tensor of shape (num_envs, 4), ordered [T, tau_x, tau_y, tau_z].
        omega_max:
            Maximum rotor speed in rad/s.

        Returns
        -------
        omega_ref, rotor_thrusts:
            Bounded rotor-speed references and their corresponding thrusts.
        """
        if wrench.ndim != 2 or wrench.shape[1] != 4:
            raise ValueError("wrench must have shape (num_envs, 4)")
        if wrench.shape[0] != self._allocation_matrix.shape[0]:
            raise ValueError("wrench batch size does not match allocator")
        if omega_max <= 0.0:
            raise ValueError("omega_max must be positive")

        dtype = wrench.dtype
        device = wrench.device
        inverse = self._inverse_allocation_matrix.to(device=device, dtype=dtype)

        # Moment-only rotor contribution. Its sum is zero by construction.
        moment_wrench = torch.zeros_like(wrench)
        moment_wrench[:, 1:] = wrench[:, 1:]
        moment_rotor = torch.bmm(
            inverse, moment_wrench.unsqueeze(-1)
        ).squeeze(-1)

        max_rotor_thrust = torch.as_tensor(
            self._thrust_coeff * float(omega_max) ** 2,
            dtype=dtype,
            device=device,
        )

        # If requested moments alone exceed the available rotor-thrust span,
        # scale all moments proportionally. This preserves their direction.
        moment_min = moment_rotor.amin(dim=1, keepdim=True)
        moment_max = moment_rotor.amax(dim=1, keepdim=True)
        span = moment_max - moment_min
        moment_scale = torch.clamp(
            max_rotor_thrust / torch.clamp(span, min=1.0e-9),
            max=1.0,
        )
        moment_rotor = moment_rotor * moment_scale

        # With the feasible moment distribution fixed, choose a shared
        # collective contribution. Clamp the requested T/4 to the interval
        # that keeps every rotor inside [0, max_rotor_thrust].
        lower = -moment_rotor.amin(dim=1, keepdim=True)
        upper = max_rotor_thrust - moment_rotor.amax(dim=1, keepdim=True)
        requested_collective_per_rotor = wrench[:, :1] / 4.0
        collective = torch.minimum(
            torch.maximum(requested_collective_per_rotor, lower),
            upper,
        )

        rotor_thrusts = torch.clamp(
            moment_rotor + collective,
            min=0.0,
            max=max_rotor_thrust,
        )
        omega_ref = torch.sqrt(
            rotor_thrusts / float(self._thrust_coeff)
        )
        return omega_ref, rotor_thrusts
