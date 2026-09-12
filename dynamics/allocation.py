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
