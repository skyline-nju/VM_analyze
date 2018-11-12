import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from read_nc import read_2, get_host_info, Snapshot_2
import matplotlib.pyplot as plt


def cal_v_perp_q(vx, vy, n_x, n_y):
    v_perp = - n_y * vx + n_x * vy
    v_perp_q = np.fft.fft2(v_perp)
    v_perp_q = np.fft.fftshift(v_perp_q)
    v_perp_q_square = np.abs(v_perp-q) ** 2
    return v_perp_q_square


def cal_rho_q(rho, rho_m=1.):
    delta_rho = rho - rho_m
    rho_q = np.fft.fft2(delta_rho)
    rho_q = np.fft.fftshift(rho_q)
    rho_q_square = np.abs(rho_q) ** 2
    return rho_q_square


def interpolate(data_mat, gl_ux, gl_uy, log_val=True, full=False):
    n = data_mat.shape[0]
    qx = np.linspace(0, 1, n, endpoint=False) - 0.5
    qy = qx

    if log_val:
        spl = RectBivariateSpline(qx, qy, np.log(data_mat.T))
    else:
        spl = RectBivariateSpline(qx, qy, data_mat.T)
    if full:
        c_q_new = spl(qx * np.abs(gl_ux), qy * np.abs(gl_uy), grid=True).T
        # c_q_new = spl(qx, qy, grid=True).T
        if log_val:
            c_q_new = np.exp(c_q_new)
        return c_q_new
    else:
        q_parallel_x = qx[n // 2:] * gl_ux
        q_parallel_y = qy[n // 2:] * gl_uy
        q_perp_x = -qx[n // 2:] * gl_uy
        q_perp_y = qy[n // 2:] * gl_ux
        c_q_parallel = spl(q_parallel_x, q_parallel_y, grid=False)
        c_q_perp = spl(q_perp_x, q_perp_y, grid=False)
        if log_val:
            c_q_parallel = np.exp(c_q_parallel)
            c_q_perp = np.exp(c_q_perp)
        return c_q_parallel, c_q_perp


if __name__ == "__main__":
    os.chdir(r"D:\data\random_torque\large_system")
    fname = "field_2400_0.18_0.000_1.0_1_host0.nc"

    # frames = read_2(fname, 20)
    # host_size, cells_per_host = get_host_info(fname)
    # nc_x = host_size[0] * cells_per_host[0]
    # Lx = int(fname.split("_")[1])
    # cell_area = (Lx / nc_x) ** 2
    # print("cell area:", cell_area)

    # for frame in frames:
    #     t, n_cell, vx_cell, vy_cell =frame
    #     n_par = np.sum(n_cell)
    #     vx_m = np.sum(vx_cell) / n_par
    #     vy_m = np.sum(vy_cell) / n_par
    #     v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
    #     n_x = vx_m / v_m
    #     n_y = vy_m / v_m

    #     mask = n_cell != 0
    #     vx_field = np.zeros_like(vx_cell)
    #     vy_field = np.zeros_like(vy_cell)
    #     vx_field[mask] = vx_cell[mask] / n_cell[mask]
    #     vy_field[mask] = vy_cell[mask] / n_cell[mask]
    #     ux_m = np.mean(vx_field)
    #     uy_m = np.mean(vy_field)
    #     print(vx_m, vy_m, np.arctan2(vy_m, vx_m) / np.pi * 180)
    #     print(ux_m, uy_m, np.arctan2(uy_m, ux_m) / np.pi * 180)
    #     break
    snap = Snapshot_2(fname)
    frames = snap.gene_frames(20, 30, 2)
    for frame in frames:
        t, num_count, vx_sum, vy_sum = frame
        print(np.mean(num_count), np.mean(vx_sum))
