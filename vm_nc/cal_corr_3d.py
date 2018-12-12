import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from read_nc import Snapshot_3
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
try:
    from corr2d import add_line
except ModuleNotFoundError:
    print("failed to find module add_line")
    sys.exit()


def cal_v_perp_q(vx, vy, vz, n_x, n_y, n_z):
    v_perp_x = vx - vx * n_x
    v_perp_y = vy - vy * n_y
    v_perp_z = vz - vz * n_z
    v_perp_qx = np.fft.fftn(v_perp_x)
    v_perp_qy = np.fft.fftn(v_perp_y)
    v_perp_qz = np.fft.fftn(v_perp_z)
    v_perp_qx = np.fft.fftshift(v_perp_qx)
    v_perp_qy = np.fft.fftshift(v_perp_qy)
    v_perp_qz = np.fft.fftshift(v_perp_qz)
    v_perp_q2 = np.abs(v_perp_qx) ** 2 + \
        np.abs(v_perp_qy) ** 2 + np.abs(v_perp_qz) ** 2
    return v_perp_q2


def cal_rho_q(rho, rho_m):
    delta_rho = rho - rho_m
    rho_q = np.fft.fftn(delta_rho)
    rho_q = np.fft.fftshift(rho_q)
    rho_q2 = np.abs(rho_q) ** 2
    return rho_q2


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


def interpolate(X, Y, Z, W, ux, uy, uz, full=False):
    n = X.size
    interp_func = RegularGridInterpolator((X, Y, Z), W.T)
    q_para = np.array(
        [[X[i] * ux, Y[i] * uy, Z[i] * uz] for i in range(n // 2, n)])
    cq_para = interp_func(q_para)
    n_ori = 6
    cq_perp = np.zeros_like(cq_para)
    perp_vecs = get_perp_vec(ux, uy, uz, n_ori)
    for n_x, n_y, n_z in perp_vecs:
        q_perp = np.array([[X[i] * n_x, Y[i] * n_y, Z[i] * n_z]
                          for i in range(n // 2, n)])
        cq_perp += interp_func(q_perp)
    cq_perp /= n_ori
    return cq_para, cq_perp


class Corr3d:
    def __init__(self, filename):
        self.snap = Snapshot_3(filename)
        self.L = self.snap.L
        print(self.L)

    def set_q(self, block_size=1):
        self.nc_z = self.snap.gl_nc[0] // block_size
        self.nc_y = self.snap.gl_nc[1] // block_size
        self.nc_x = self.snap.gl_nc[2] // block_size
        dx = self.L / self.nc_x
        dy = self.L / self.nc_y
        dz = self.L / self.nc_z
        d_qx = d_qy = d_qz = np.pi * 2 / self.L
        self.cell_vol = dx * dy * dz
        self.qx = np.linspace(
            -self.nc_x/2, self.nc_x/2, self.nc_x, False) * d_qx
        self.qy = np.linspace(
            -self.nc_y/2, self.nc_y/2, self.nc_y, False) * d_qy
        self.qz = np.linspace(
            -self.nc_z/2, self.nc_z/2, self.nc_z, False) * d_qz
        
    def time_average(self, first_frame=0, last_frame=None, block_size=1):
        self.set_q(block_size)
        print("cell volume =", self.cell_vol)
        frames = self.snap.gene_frames(first_frame, last_frame, block_size)
        rho_q_sum = np.zeros((self.nc_z, self.nc_y, self.nc_x))
        v_q_sum = np.zeros_like(rho_q_sum)
        count = 0
        n_x_sum = n_y_sum = n_z_sum = 0.
        for frame in frames:
            t, n_cell, vx_cell, vy_cell, vz_cell = frame
            mask = n_cell != 0
            vx = np.zeros_like(vx_cell)
            vy = np.zeros_like(vx_cell)
            vz = np.zeros_like(vx_cell)
            vx[mask] = vx_cell[mask] / n_cell[mask]
            vy[mask] = vy_cell[mask] / n_cell[mask]
            vz[mask] = vz_cell[mask] / n_cell[mask]
            vx_m = np.mean(vx)
            vy_m = np.mean(vy)
            vz_m = np.mean(vz)
            v_m = np.sqrt(vx_m ** 2 + vy_m ** 2 + vz_m ** 2)
            n_x, n_y, n_z = vx_m / v_m, vy_m / v_m, vz_m / v_m
            print("t =", t, "vm =", n_x, n_y, n_z)
            n_x_sum += n_x
            n_y_sum += n_y
            n_z_sum += n_z
            rho = n_cell / self.cell_vol
            rho_q_sum += cal_rho_q(rho, 1.) / n_cell.size
            v_q_sum += cal_v_perp_q(vx, vy, vz, n_x, n_y, n_z) / n_cell.size
            count += 1
        n_x_mean = n_x_sum / count
        n_y_mean = n_y_sum / count
        n_z_mean = n_z_sum / count
        rho_q_mean = rho_q_sum / count
        v_q_mean = v_q_sum / count

        outfile = r"..\Cq_%d_%.2f_%.3f_%d.npz" % (self.snap.L, self.snap.eta,
                                                  self.snap.eps, self.snap.seed)
        np.savez(outfile, rho_q=rho_q_mean, v_q=v_q_mean,
                 n_x=n_x_mean, n_y=n_y_mean, n_z=n_z_mean,
                 qx=self.qx, qy=self.qy, qz=self.qz)


if __name__ == "__main__":
    os.chdir(r"E:\data\vm3d\field")
    # fname = "field_240_0.20_0.120_1.0_21.nc"
    # corr = Corr3d(fname)
    # corr.time_average(20)

    fname = r"..\Cq_240_0.20_0.120_21.npz"
    npzfile = np.load(fname)
    rho_q = npzfile["rho_q"]
    v_q = npzfile["v_q"]
    n_x = npzfile["n_x"]
    n_y = npzfile["n_y"]
    n_z = npzfile["n_z"]
    qx = npzfile["qx"]
    qy = npzfile["qy"]
    qz = npzfile["qz"]
    v_q_para, v_q_perp = interpolate(qx, qy, qz, v_q, n_x, n_y, n_z)
    rho_q_para, rho_q_perp = interpolate(qx, qy, qz, rho_q, n_x, n_y, n_z)
    q_half = qx[qx.size//2:]
    # plt.plot(q_half, v_q_para, "o")
    # plt.plot(q_half, v_q_perp, "s")
    plt.plot(q_half, rho_q_para, "o")
    plt.plot(q_half, rho_q_perp, "s")
    plt.xscale("log")
    plt.yscale("log")
    add_line.add_line(plt.gca(), 0, 1, 1, -2, scale="log")
    plt.show()
    plt.close()
