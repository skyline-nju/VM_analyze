import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from read_nc import Snapshot_2
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
try:
    from corr2d import add_line
except ModuleNotFoundError:
    print("failed to find module add_line")
    sys.exit()

def cal_v_perp_q(vx, vy, n_x, n_y):
    v_perp = - n_y * vx + n_x * vy
    v_perp_q = np.fft.fft2(v_perp)
    v_perp_q = np.fft.fftshift(v_perp_q)
    v_perp_q_square = np.abs(v_perp_q) ** 2
    return v_perp_q_square


def cal_rho_q(rho, rho_m=1.):
    delta_rho = rho - rho_m
    rho_q = np.fft.fft2(delta_rho)
    rho_q = np.fft.fftshift(rho_q)
    rho_q_square = np.abs(rho_q) ** 2
    return rho_q_square


def interpolate(X, Y, Z, ux, uy, log_z=True, full=False):
    n = X.size
    if log_z:
        spl = RectBivariateSpline(X, Y, np.log(Z.T))
    else:
        spl = RectBivariateSpline(X, Y, Z.T)
    if full:
        Z_new = spl(X * np.abs(ux), Y * np.abs(uy), grid=True).T
        if log_z:
            Z_new = np.exp(Z_new)
        return Z_new
    else:
        x_para = X[n//2:] * ux
        y_para = Y[n//2:] * uy
        x_perp = -X[n//2:] * uy
        y_perp = Y[n//2:] * ux
        Z_para = spl(x_para, y_para, grid=False)
        Z_perp = spl(x_perp, y_perp, grid=False)
        if log_z:
            Z_para = np.exp(Z_para)
            Z_perp = np.exp(Z_perp)
        return Z_para, Z_perp


class Corr2d:
    def __init__(self, filename):
        self.snap = Snapshot_2(filename)
        self.L = self.snap.L
    
    def set_q(self, block_size):
        self.nc_y = self.snap.gl_nc[0] // block_size
        self.nc_x = self.snap.gl_nc[1] // block_size
        dx = self.L / self.nc_x
        dy = self.L / self.nc_y
        d_qx = np.pi * 2 / self.L
        d_qy = np.pi * 2 / self.L
        self.cell_area = dx * dy
        self.qx = np.linspace(
            -self.nc_x/2, self.nc_x/2, self.nc_x, False) * d_qx
        self.qy = np.linspace(
            -self.nc_y/2, self.nc_y/2, self.nc_y, False) * d_qy

    
    def time_average(self, first_frame=0, last_frame=None, block_size=1):
        self.set_q(block_size)
        print("cell area =", self.cell_area)
        frames = self.snap.gene_frames(first_frame, last_frame, block_size)
        rho_q_sum = np.zeros((self.nc_y, self.nc_x))
        v_q_sum = np.zeros((self.nc_y, self.nc_x))
        count = 0
        n_x_sum = 0
        n_y_sum = 0
        for frame in frames:
            t, n_cell, vx_cell, vy_cell = frame
            mask = n_cell != 0
            vx = np.zeros_like(vx_cell)
            vy = np.zeros_like(vy_cell)
            vx[mask] = vx_cell[mask] / n_cell[mask]
            vy[mask] = vy_cell[mask] / n_cell[mask]
            vx_m = np.mean(vx)
            vy_m = np.mean(vy)
            v_m = np.sqrt(vx_m ** 2 + vy_m ** 2)
            print("t =", t, "theta =", np.arctan2(vy_m, vx_m) * 180 / np.pi)
            n_x = vx_m / v_m
            n_y = vy_m / v_m
            n_x_sum += n_x
            n_y_sum += n_y
            rho = n_cell / self.cell_area
            rho_q_sum += cal_rho_q(rho, 1.) / n_cell.size
            v_q_sum += cal_v_perp_q(vx, vy, n_x, n_y) / n_cell.size
            count += 1
        n_x_mean = n_x_sum / count
        n_y_mean = n_y_sum / count
        rho_q_mean = rho_q_sum / count
        v_q_mean = v_q_sum / count

        outfile = r"Cq_%d_%.2f_%.3f_%d.npz" % (self.snap.L, self.snap.eta,
                                               self.snap.eps, self.snap.seed)
        np.savez(outfile, rho_q=rho_q_mean, v_q=v_q_mean,
                 n_x=n_x_mean, n_y=n_y_mean, qx=self.qx, qy=self.qy)


def plot_q2vq2(v_q, qx, qy, n_x, n_y, q_min, q_max):
    theta = []
    y = []
    for j in range(qy.size):
        for i in range(qx.size):
            q = np.sqrt(qy[j] ** 2 + qx[i] ** 2)
            if q > q_min and q < q_max:
                c = qx[i] * n_x + qy[j] * n_y
                s = qy[j] * n_x - qx[i] * n_y
                theta.append(np.arctan2(s, c))
                y.append(q ** 2 * v_q[j, i])
    plt.plot(theta, y, "o")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir(r"E:\data\random_torque\large_system")
    # fname = "field_4800_0.18_0.020_1.0_1_host0.nc"
    # corr = Corr2d(fname)
    # corr.time_average(first_frame=100, last_frame=101)

    fname = "Cq_4800_0.18_0.020_1.npz"
    npzfile = np.load(fname)
    rho_q = npzfile['rho_q']
    v_q = npzfile["v_q"]
    n_x = npzfile['n_x']
    n_y = npzfile['n_y']
    qx = npzfile["qx"]
    qy = npzfile['qy']
    # v_q_para, v_q_perp = interpolate(qx, qy, v_q, n_x, n_y, log_z=False)
    # rho_q_para, rho_q_perp = interpolate(qx, qy, rho_q, n_x, n_y, log_z=False)
    q_half = qx[qx.size//2:]
    print(q_half[1], q_half[-1])
    # plt.plot(q_half, v_q_para, "o")
    # plt.plot(q_half, v_q_perp, "s")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(0.01, 200)
    # plt.xlim(6e-3)
    # add_line.add_line(plt.gca(), 0, 1, 1, -2, scale="log")
    # add_line.add_line(plt.gca(), 0, 0.8, 1, -2, scale="log")

    # plt.show()
    # plt.close()
    plot_q2vq2(v_q, qx, qy, n_x, n_y, 6e-3, 2e-2)
