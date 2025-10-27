#!/usr/bin/env python3
# This is the robot-side code, may not be executed on other robots.

import torch
from torch import sin, cos, acos, atan, asin, pi, sqrt
import numpy as np

# Class for kinematic calculations of a robot leg with three links
class Kinematic:
    def __init__(self, l1, l2, l3, device='cpu'):
        # Initialize link lengths (in meters) and computation device
        self.l1 = l1  # Length of first link
        self.l2 = l2  # Length of second link
        self.l3 = l3  # Length of third link
        self.device = device  # Device for tensor operations (default: CPU)

    def ForwardKin(self, joints, pos: torch.Tensor):
        """Compute forward kinematics for a batch of joint angles.
        Args:
            joints: Tensor of shape [batch_size, 3] with joint angles (q1, q2, q3)
            pos: Tensor of shape [batch_size, 3] to store end-effector positions (x, y, z)
        """
        q1 = joints[:, 0]  # First joint angle (base rotation)
        q2 = joints[:, 1]  # Second joint angle (thigh)
        q3 = joints[:, 2]  # Third joint angle (knee)
        # Compute x, y, z coordinates of end-effector using forward kinematics
        pos[:, 0] = self.l1 * cos(q1) + self.l2 * cos(q1) * cos(q2) + self.l3 * cos(q1) * cos(q2 + q3)
        pos[:, 1] = self.l1 * sin(q1) + self.l2 * sin(q1) * cos(q2) + self.l3 * sin(q1) * cos(q2 + q3)
        pos[:, 2] = self.l2 * sin(q2) + self.l3 * sin(q2 + q3)

    def DampInvJac(self, joints):
        """Compute damped inverse Jacobian for a batch of joint angles.
        Args:
            joints: Tensor of shape [batch_size, 3] with joint angles (q1, q2, q3)
        Returns:
            Tensor of shape [batch_size, 3, 3] with damped inverse Jacobians
        """
        # Initialize Jacobian tensor
        Jac = torch.zeros(joints.shape[0], 3, 3, dtype=torch.float32, device=self.device)
        q1 = joints[:, 0]
        q2 = joints[:, 1]
        q3 = joints[:, 2]
        # Compute Jacobian elements (partial derivatives of position w.r.t. joint angles)
        Jac[:, 0, 0] = -self.l1 * sin(q1) - (self.l2 * cos(q2) + self.l3 * cos(q2 + q3)) * sin(q1)
        Jac[:, 1, 0] = self.l1 * cos(q1) + (self.l2 * cos(q2) + self.l3 * cos(q2 + q3)) * cos(q1)
        Jac[:, 2, 0] = 0
        Jac[:, 0, 1] = -(self.l2 * sin(q2) + self.l3 * sin(q2 + q3)) * cos(q1)
        Jac[:, 1, 1] = -(self.l2 * sin(q2) + self.l3 * sin(q2 + q3)) * sin(q1)
        Jac[:, 2, 1] = self.l2 * cos(q2) + self.l3 * cos(q2 + q3)
        Jac[:, 0, 2] = -self.l3 * cos(q1) * sin(q2 + q3)
        Jac[:, 1, 2] = -self.l3 * sin(q1) * sin(q2 + q3)
        Jac[:, 2, 2] = self.l3 * cos(q2 + q3)
        # Compute damped inverse: J^T * (J * J^T + λI)^(-1), where λ = 0.0001
        JJT = Jac @ Jac.transpose(1, 2) + torch.eye(3, 3, dtype=torch.float32, device=self.device) * 0.0001
        return Jac.transpose(1, 2) @ torch.inverse(JJT)

    def InverseKin1(self, pos: torch.Tensor, joints_cur: torch.Tensor):
        """Compute inverse kinematics iteratively using damped Jacobian.
        Args:
            pos: Tensor of shape [batch_size, 3] with target end-effector positions
            joints_cur: Tensor of shape [batch_size, 3] with current joint angles, modified in-place
        """
        # Initialize current position tensor
        pos_cur = torch.zeros_like(pos, dtype=torch.float32, device=self.device)
        self.ForwardKin(joints_cur, pos_cur)
        # Iterate up to 1000 times or until convergence
        for _ in range(1000):
            # Compute position error norm for each leg
            diff_norm = torch.linalg.norm(pos - pos_cur, dim=1)
            indices = torch.where(diff_norm < 0.005)[0]
            # Check if all legs have converged
            if indices.numel() == pos.shape[0]:
                return
            # Compute damped inverse Jacobian and update joint angles
            damp_inv_jacs = self.DampInvJac(joints_cur)
            joints_cur.add_(0.005 * (damp_inv_jacs @ ((pos - pos_cur).unsqueeze(-1))).squeeze(-1))
            self.ForwardKin(joints_cur, pos_cur)
        # Check for convergence failure
        diff_norm = torch.linalg.norm(pos - pos_cur, dim=1)
        indices = torch.where(diff_norm > 0.005)[0]
        if indices.numel() > 0:
            print(f"IK failed, indices: {indices}, pos_cur[indices]:\n{pos_cur[indices]}")

    def InverseKin2(self, pos: torch.Tensor, joints_cur: torch.Tensor):
        """Compute inverse kinematics analytically, selecting angles closest to current.
        Args:
            pos: Tensor of shape [batch_size, 3] with target end-effector positions
            joints_cur: Tensor of shape [batch_size, 3] with current joint angles, modified in-place
        """
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        # Compute possible q1 angles (base rotation)
        q1 = atan(y / x)
        q1_possible = torch.stack([q1, q1 + pi, q1 - pi], dim=0)
        joints_cur[:, 0] = self._SelectCloestAngles(q1_possible, joints_cur[:, 0])
        # Compute q3 using cosine law
        q3 = acos(((cos(joints_cur[:, 0]) * x + sin(joints_cur[:, 0]) * y - self.l1)**2 + z**2 - self.l3**2 - self.l2**2) / (2 * self.l2 * self.l3))
        q3_possible = torch.stack([q3, -q3], dim=0)
        joints_cur[:, 2] = self._SelectCloestAngles(q3_possible, joints_cur[:, 2])
        # Compute q2 using two approaches and combine possible solutions
        q2_back = atan((self.l3 * sin(joints_cur[:, 2])) / (self.l3 * cos(joints_cur[:, 2]) + self.l2))
        q2_front = asin(z / sqrt((self.l3 * cos(joints_cur[:, 2]) + self.l2)**2 + (self.l3 * sin(joints_cur[:, 2]))**2))
        q2_b_possible = torch.stack([q2_back, q2_back + pi, q2_back - pi], dim=0)
        q2_f_possible = torch.stack([q2_front, -q2_front + pi, -q2_front - pi], dim=0)
        # Combine all possible q2 solutions
        q2_possible = (q2_f_possible.unsqueeze(0) - q2_b_possible.unsqueeze(1)).reshape(-1, pos.shape[0])
        joints_cur[:, 1] = self._SelectCloestAngles(q2_possible, joints_cur[:, 1])

    def _SelectCloestAngles(self, possible_angles, current_angles):
        """Select angles closest to current angles from possible solutions.
        Args:
            possible_angles: Tensor of shape [possible_num, batch_size] with candidate angles
            current_angles: Tensor of shape [batch_size] with current angles
        Returns:
            Tensor of shape [batch_size] with selected angles
        """
        min_index = torch.abs(possible_angles - current_angles).argmin(dim=0)
        selected_angles = possible_angles[min_index, torch.arange(current_angles.shape[0])]
        return selected_angles