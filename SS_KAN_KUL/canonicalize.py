"""Helpers for canonicalizing a learned 2-state state-space model.

The PyPI package named ``canonicalize`` is unrelated to this notebook.
This module provides the small set of utilities the notebook expects.
"""

from __future__ import annotations

import numpy as np
import torch


def canonicalize_position_velocity(A, B, C, D, dt: float = 1.0):
    """Build a position/velocity-like similarity transform for a 2-state model.

    The first basis vector is aligned with the output matrix ``C`` so that the
    transformed output becomes ``[1, 0]``. The second basis vector is chosen
    from a finite-difference approximation of the output dynamics.
    """

    A_t = torch.as_tensor(A, dtype=torch.float32).detach().cpu()
    B_t = torch.as_tensor(B, dtype=torch.float32).detach().cpu()
    C_t = torch.as_tensor(C, dtype=torch.float32).detach().cpu()
    D_t = torch.as_tensor(D, dtype=torch.float32).detach().cpu()

    if C_t.ndim != 2 or C_t.shape[0] != 1:
        raise ValueError("canonicalize_position_velocity expects a single-output model")
    if A_t.shape[0] != 2 or A_t.shape[1] != 2:
        raise ValueError("canonicalize_position_velocity currently expects a 2-state model")

    row0 = C_t.squeeze(0)
    row1 = (C_t @ A_t - C_t).squeeze(0) / float(dt)

    T = torch.stack([row0, row1], dim=0)
    if torch.abs(torch.linalg.det(T)) < 1e-8:
        T = torch.stack([row0, torch.tensor([0.0, 1.0], dtype=torch.float32)], dim=0)

    T_inv = torch.linalg.inv(T)

    A_x = T @ A_t @ T_inv
    B_x = T @ B_t
    C_x = C_t @ T_inv
    D_x = D_t.clone()

    return T.numpy(), A_x, B_x, C_x, D_x


def transform_trajectory_and_nonlinearity(model, z_traj, u_traj, T_np):
    """Transform a learned-state trajectory and its nonlinear correction."""

    T = torch.as_tensor(T_np, dtype=z_traj.dtype, device=z_traj.device)
    T_inv = torch.linalg.inv(T)

    x_traj = z_traj @ T_inv.T
    f_z = model.state_kan_model(state=z_traj, u=u_traj)
    f_x = f_z @ T_inv.T
    return x_traj, f_x


def fit_cubic(x, y):
    """Least-squares fit of y ≈ a3*x^3 + a2*x^2 + a1*x + a0."""

    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y, dtype=x_t.dtype, device=x_t.device).reshape(-1, 1)
    design = torch.stack([x_t**3, x_t**2, x_t, torch.ones_like(x_t)], dim=1)
    coeffs = torch.linalg.lstsq(design, y_t).solution.squeeze()
    fit = design @ coeffs.reshape(-1, 1)
    return coeffs, fit.squeeze()