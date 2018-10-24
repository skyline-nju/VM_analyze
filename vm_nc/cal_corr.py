import numpy as np
import os
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from read_nc import read_2, get_host_info, read_3
import matplotlib.pyplot as plt


def cal_corr_v_q_2(vx, vy, n_x, n_y):
    v_perp = - n_y * vx + n_x * vy
    v_perp_q = np.fft.fft2(v_perp)
    v_perp_q = np.fft.fftshift(v_perp_q)
    v_perp_q_power = np.abs(v_perp_q) ** 2
    return v_perp_q_power


def cal_corr_v_q_3(vx, vy, vz, n_x, n_y, n_z):
    vx_para = vx * n_x
    vy_para = vy * n_y
    vz_para = vz * n_z
    v_perp_x = vx - vx_para
    v_perp_y = vy - vy_para
    v_perp_z = vz - vz_para
    v_perp_qx = np.fft.fftn(v_perp_x)
    v_perp_qy = np.fft.fftn(v_perp_y)
    v_perp_qz = np.fft.fftn(v_perp_z)
    v_perp_qx = np.fft.fftshift(v_perp_qx)
    v_perp_qy = np.fft.fftshift(v_perp_qy)
    v_perp_qz = np.fft.fftshift(v_perp_qz)
    v_perp_q2 = np.abs(v_perp_qx) ** 2 + \
        np.abs(v_perp_qy) ** 2 + np.abs(v_perp_qz) ** 2
    return v_perp_q2


def cal_corr_rho_q_2(rho, rho_m):
    delta_rho = rho - rho_m
    c_rho_q = np.fft.fft2(delta_rho)
    c_rho_q = np.fft.fftshift(c_rho_q)
    c_rho_q = np.abs(c_rho_q) ** 2
    return c_rho_q


def cal_corr_rho_q_3(rho, rho_m):
    delta_rho = rho - rho_m
    c_rho_q = np.fft.fftn(delta_rho)
    c_rho_q = np.fft.fftshift(c_rho_q)
    c_rho_q = np.abs(c_rho_q) ** 2
    return c_rho_q


def interpolate(data_mat, gl_ux, gl_uy, log_val=True, full=False):
    n = data_mat.shape[0]
    qx = np.linspace(0, 1, n, endpoint=False) - 0.5
    qy = qx

    if data_mat.ndim == 2:
        if log_val:
            spl = RectBivariateSpline(qx, qy, np.log(data_mat.T))
        else:
            spl = RectBivariateSpline(qx, qy, data_mat.T)
        if full:
            c_q_new = spl(qx * np.abs(gl_ux), qy * np.abs(gl_uy), grid=True).T
            # c_q_new = spl(qx, qy, grid=True).T
        else:
            q_parallel_x = qx[n // 2:] * gl_ux
            q_parallel_y = qy[n // 2:] * gl_uy
            q_perp_x = -qx[n // 2:] * gl_uy
            q_perp_y = qy[n // 2:] * gl_ux
            c_q_parallel = spl(q_parallel_x, q_parallel_y, grid=False)
            c_q_perp = spl(q_perp_x, q_perp_y, grid=False)
    if full:
        if log_val:
            c_q_new = np.exp(c_q_new)
        return c_q_new
    else:
        if log_val:
            c_q_parallel = np.exp(c_q_parallel)
            c_q_perp = np.exp(c_q_perp)
        return c_q_parallel, c_q_perp


def interpolate_circle(data_mat, gl_ux, gl_uy, r_arr, n_theta=60):
    n = data_mat.shape[0]
    q = np.linspace(0, 1, n, endpoint=False) - 0.5
    theta = np.linspace(-0.5, 0.5, n_theta, endpoint=False) * np.pi * 2
    theta_q = theta + np.arctan2(gl_uy, gl_ux)
    qx = np.cos(theta_q) * r_arr[0]
    qy = np.sin(theta_q) * r_arr[0]
    spl = RectBivariateSpline(q, q, np.log(data_mat.T))
    cq2 = spl(qx, qy, grid=False)
    return theta, cq2


def get_perp_vec(n_x, n_y, n_z, count):
    s = np.sqrt(n_x ** 2 + n_y ** 2)
    c = n_z
    rot_axis_x = -n_y / s
    rot_axis_y = n_x / s
    rot_axis_z = 0.

    for i in range(count):
        theta = i / count * np.pi
        ux = np.cos(theta)
        uy = np.sin(theta)
        uz = 0
        bxx = rot_axis_x ** 2 * (1 - c)
        bxy = rot_axis_x * rot_axis_y * (1 - c)
        bxz = rot_axis_x * rot_axis_z * (1 - c)
        byy = rot_axis_y ** 2 * (1 - c)
        byz = rot_axis_y * rot_axis_z * (1 - c)
        bzz = rot_axis_z ** 2 * (1 - c)
        x_new = (bxx + c) * ux + (bxy - rot_axis_z * s) * uy \
            + (bxz + rot_axis_y * s) * uz
        y_new = (bxy + rot_axis_z * s) * ux + (byy + c) * uy \
            + (byz - rot_axis_x * s) * uz
        z_new = (bxz - rot_axis_y * s) * ux + (byz + rot_axis_x * s) * uy \
            + (bzz + c) * uz
        yield x_new, y_new, z_new


def interpolate_3(mat, n_x, n_y, n_z):
    qx = np.linspace(0, 1, mat.shape[0], endpoint=False) - 0.5
    qy = np.linspace(0, 1, mat.shape[1], endpoint=False) - 0.5
    qz = np.linspace(0, 1, mat.shape[2], endpoint=False) - 0.5
    interp_func = RegularGridInterpolator((qx, qy, qz), mat.T)
    q_para = np.array([[qx[i] * n_x, qy[i] * n_y, qz[i] * n_z] for i in range(
        qx.size // 2, qx.size)])
    c_q_para = interp_func(q_para)
    count = 6
    c_q_perp = np.zeros_like(c_q_para)
    perp_vecs = get_perp_vec(n_x, n_y, n_z, count)
    for ux, uy, uz in perp_vecs:
        q_perp = np.array([[qx[i] * ux, qy[i] * uy, qz[i] * uz]
                           for i in range(qx.size // 2, qx.size)])
        c_q_perp += interp_func(q_perp)
    c_q_perp /= count
    return c_q_para, c_q_perp


def cal_corr_q_2(file0, first_frame=20):
    frames = read_2(file0, first_frame)
    host_size, cells_per_host = get_host_info(file0)
    nc_x = host_size[0] * cells_per_host[0]
    Lx = int(file0.split("_")[1])
    cell_area = (Lx / nc_x) ** 2
    c_v_para, c_v_perp, c_rho_para, c_rho_perp = np.zeros((4, nc_x // 2))
    q = np.linspace(0, 0.5, nc_x // 2, endpoint=False)
    count = 0
    for frame in frames:
        t, n_cell, vx_cell, vy_cell = frame
        print("t =", t)
        tot_par = np.sum(n_cell)
        vx_m = np.sum(vx_cell) / tot_par
        vy_m = np.sum(vy_cell) / tot_par
        v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
        n_x = vx_m / v_m
        n_y = vy_m / v_m
        mask = n_cell != 0
        vx_field = np.zeros_like(vx_cell)
        vy_field = np.zeros_like(vy_cell)
        vx_field[mask] = vx_cell[mask] / n_cell[mask]
        vy_field[mask] = vy_cell[mask] / n_cell[mask]
        rho = np.array([i / cell_area for i in n_cell])
        c_rho_q2 = cal_corr_rho_q_2(rho, 1) / n_cell.size
        c_para, c_perp = interpolate(c_rho_q2, n_x, n_y)
        c_rho_para += c_para
        c_rho_perp += c_perp
        c_v_q2 = cal_corr_v_q_2(vx_field, vy_field, n_x, n_y) / n_cell.size
        c_para, c_perp = interpolate(c_v_q2, n_x, n_y)
        c_v_para += c_para
        c_v_perp += c_perp
        count += 1
    c_v_para /= count
    c_v_perp /= count
    c_rho_para /= count
    c_rho_perp /= count
    outfile = "cq" + file0.replace("field", "").replace("_host0.nc", ".dat")
    with open(outfile, "w") as f:
        lines = ["%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
            q[i], c_v_para[i], c_v_perp[i], c_rho_para[i], c_rho_perp[i])
            for i in range(q.size)]
        f.writelines(lines)


def plot_theta_q_2(file0, first_frame=20):
    frames = read_2(file0, first_frame)
    host_size, cells_per_host = get_host_info(file0)
    nc_x = host_size[0] * cells_per_host[0]
    # Lx = int(file0.split("_")[1])
    # cell_area = (Lx / nc_x) ** 2
    c_v_para, c_v_perp, c_rho_para, c_rho_perp = np.zeros((4, nc_x // 2))
    count = 0
    for frame in frames:
        t, n_cell, vx_cell, vy_cell = frame
        print("t =", t)
        tot_par = np.sum(n_cell)
        vx_m = np.sum(vx_cell) / tot_par
        vy_m = np.sum(vy_cell) / tot_par
        v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
        n_x = vx_m / v_m
        n_y = vy_m / v_m
        mask = n_cell != 0
        vx_field = np.zeros_like(vx_cell)
        vy_field = np.zeros_like(vy_cell)
        vx_field[mask] = vx_cell[mask] / n_cell[mask]
        vy_field[mask] = vy_cell[mask] / n_cell[mask]
        c_v_q2 = cal_corr_v_q_2(vx_field, vy_field, n_x, n_y) / n_cell.size
        tmp = np.linspace(-np.pi, np.pi, nc_x, endpoint=False)
        qx, qy = np.meshgrid(tmp, tmp)
        q2 = qx ** 2 + qy ** 2
        theta_q = np.arctan2(qy, qx) - np.arctan2(n_y, n_x)
        y = q2 * c_v_q2
        plt.plot(theta_q, y, "o")
        plt.show()
        plt.close()
        count += 1


def cal_corr_q_3(file0, first_frame=20):
    frames = read_3(file0, first_frame)
    host_size, cells_per_host = get_host_info(file0)
    nc_x = host_size[0] * cells_per_host[0]
    Lx = int(file0.split("_")[1])
    cell_area = (Lx / nc_x) ** 2
    c_v_para, c_v_perp, c_rho_para, c_rho_perp = np.zeros((4, nc_x // 2))
    q = np.linspace(0, 0.5, nc_x // 2, endpoint=False)
    c_rho_para = np.zeros_like(q)
    c_rho_perp = np.zeros_like(q)
    c_v_para = np.zeros_like(q)
    c_v_perp = np.zeros_like(q)
    count = 0
    for frame in frames:
        t, n_cell, vx_cell, vy_cell, vz_cell = frame
        print("t =", t)
        tot_par = np.sum(n_cell)
        vx_m = np.sum(vx_cell) / tot_par
        vy_m = np.sum(vy_cell) / tot_par
        vz_m = np.sum(vz_cell) / tot_par
        v_m = np.sqrt(vx_m ** 2 + vy_m ** 2 + vz_m ** 2)
        n_x = vx_m / v_m
        n_y = vy_m / v_m
        n_z = vz_m / v_m
        mask = n_cell != 0
        vx_field = np.zeros_like(vx_cell)
        vy_field = np.zeros_like(vy_cell)
        vz_field = np.zeros_like(vz_cell)
        vx_field[mask] = vx_cell[mask] / n_cell[mask]
        vy_field[mask] = vy_cell[mask] / n_cell[mask]
        vz_field[mask] = vz_cell[mask] / n_cell[mask]
        rho = np.array([i / cell_area for i in n_cell])
        c_rho_q = cal_corr_rho_q_3(rho, 1) / n_cell.size
        c_para, c_perp = interpolate_3(c_rho_q, n_x, n_y, n_z)
        c_rho_para += c_para
        c_rho_perp += c_perp
        c_v_q = cal_corr_v_q_3(vx_field, vy_field, vz_field,
                               n_x, n_y, n_z) / n_cell.size
        c_para, c_perp = interpolate_3(c_v_q, n_x, n_y, n_z)
        c_v_para += c_para
        c_v_perp += c_perp
        count += 1
    c_v_para /= count
    c_v_perp /= count
    c_rho_para /= count
    c_rho_perp /= count
    # plt.loglog(q, c_rho_perp)
    # plt.show()
    # plt.close()
    outfile = "cq" + file0.replace("field", "").replace(".nc", ".dat")
    with open(outfile, "w") as f:
        lines = ["%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
            q[i], c_v_para[i], c_v_perp[i], c_rho_para[i], c_rho_perp[i])
            for i in range(q.size)]
        f.writelines(lines)


def plot_corr_q_2(file0, first_frame=20, frame_count=1):
    frames = read_2(file0, first_frame)
    host_size, cells_per_host = get_host_info(file0)
    n_cells_x = host_size[0] * cells_per_host[0]
    Lx = int(file0.split("_")[1])
    cell_area = (Lx / n_cells_x) ** 2
    print("cell area =", cell_area)
    c_v_para, c_v_perp, c_rho_para, c_rho_perp = np.zeros((4, n_cells_x // 2))
    rho_q_2_accum1 = np.zeros((n_cells_x, n_cells_x))
    rho_q_2_accum2 = np.zeros((n_cells_x, n_cells_x))
    for i, frame in enumerate(frames):
        if i >= frame_count:
            break
        t, num, vx, vy = frame
        tot_par = np.sum(num)
        vx_m = np.sum(vx) / tot_par
        vy_m = np.sum(vy) / tot_par
        v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
        gl_ux = vx_m / v_m
        gl_uy = vy_m / v_m
        print("t =", t, "ux =", gl_ux, "uy =", gl_uy)
        mask = num != 0
        rho_field = np.zeros((num.shape[0], num.shape[1]))
        rho_field[mask] = num[mask] / cell_area
        rho_q_2 = cal_corr_rho_q_2(rho_field, 0) / rho_field.size
        rho_q_2_new = interpolate(rho_q_2, gl_ux, gl_uy, True, True)
        rho_q_2_accum1 += rho_q_2
        rho_q_2_accum2 += rho_q_2_new
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    rho_q_2_accum1 /= i
    rho_q_2_accum2 /= i
    im1 = ax1.imshow(np.log10(rho_q_2_accum1), origin="lower", vmin=0, vmax=5)
    im2 = ax2.imshow(np.log10(rho_q_2_accum2), origin="lower", vmin=0, vmax=5)
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.show()
    plt.close()


def find_theta_c_2(file0, first_frame=40, frame_count=1):
    frames = read_2(file0, first_frame)
    host_size, cells_per_host = get_host_info(file0)
    nx = host_size[0] * cells_per_host[0]
    print("nx =", nx)
    Lx = int(file0.split("_")[1])
    cell_area = (Lx / nx) ** 2
    n_theta = 60
    q2_cq2 = np.zeros(n_theta)
    for i, frame in enumerate(frames):
        t, num, vx, vy = frame
        tot_par = np.sum(num)
        vx_m = np.sum(vx) / tot_par
        vy_m = np.sum(vy) / tot_par
        v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
        gl_ux = vx_m / v_m
        gl_uy = vy_m / v_m
        rho_field = num / cell_area
        # rho_q_2 = cal_corr_rho_q_2(rho_field, 0) / rho_field.size
        mask = num != 0
        vx_field = np.zeros_like(vx)
        vy_field = np.zeros_like(vy)
        vx_field[mask] = vx[mask] / num[mask]
        vy_field[mask] = vy[mask] / num[mask]
        v_q_2 = cal_corr_v_q_2(vx_field, vy_field, gl_ux, gl_uy)
        # qx = np.linspace(0, 1, nx, endpoint=False) - 0.5
        # qy = qx
        # k = 10
        # theta_q = np.zeros((k * 2, k * 2))
        # theta_v = np.arctan2(gl_uy, gl_ux)
        # q2_cq2 = np.zeros_like(theta_q)
        # for iy in range(-k, k):
        #     my_qy = qy[nx // 2 + iy]
        #     for ix in range(-k, k):
        #         my_qx = qx[nx // 2 + ix]
        #         q2_cq2[iy + k, ix + k] = (my_qx ** 2 + my_qy ** 2) * \
        #             rho_q_2[nx // 2 + iy, nx // 2 + ix]
        #         theta_q[iy + k, ix + k] = np.arctan2(my_qy, my_qx) - theta_v
        # theta_q[theta_q > np.pi] -= np.pi * 2
        # theta_q[theta_q < -np.pi] += np.pi * 2
        q = 0.001
        theta, cq2 = interpolate_circle(v_q_2, gl_ux, gl_uy, [q])
        q2_cq2 += cq2 * q ** 2
    plt.plot(theta, q2_cq2 / i, "o")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir(r"D:\data\random_torque\large_system")
    fname = "field_2400_0.18_0.000_1.0_1_host0.nc"
    # cal_corr_q_2(fname, 50)
    # plot_theta_q_2(fname)
    # plot_corr_q_2(fname, 60, 1)
    find_theta_c_2(fname, 40, 1)

    # os.chdir(r"D:\data\vm3d")
    # fname = "field_240_0.20_0.000_1.0_12.nc"
    # cal_corr_q_3(fname)
