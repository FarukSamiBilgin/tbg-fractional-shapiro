"""Plateau detection and differential-resistance helpers."""
import numpy as np


def step_width(v_row, idc_grid, v_target, tol):
    """Width of a Shapiro plateau: measure of i_dc points locked to v_target."""
    mask = np.abs(v_row - v_target) < tol
    if not mask.any():
        return 0.0
    di = idc_grid[1] - idc_grid[0]
    return mask.sum() * di


def width_curve(v_map, idc_grid, v_target, tol):
    """step_width along axis 0 of a (n_iac, n_idc) voltage map."""
    return np.array([step_width(row, idc_grid, v_target, tol) for row in v_map])


def dvdi(v_map, idc_grid):
    """Differential resistance dV/dI along the i_dc axis (last axis)."""
    return np.gradient(v_map, idc_grid, axis=-1)
