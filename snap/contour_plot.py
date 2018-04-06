import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import os
import sys


def plot_density(rho,
                 Lx, Ly,
                 rho_max=10,
                 ax=None,
                 cmap="jet",
                 log_scale=True,
                 gamma=0.7):
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False
    nrows, ncols = rho.shape
    x = np.linspace(0.5 * Lx / ncols, Lx - 0.5 * Lx / ncols, ncols)
    y = np.linspace(0.5 * Ly / ncols, Ly - 0.5 * Ly / nrows, nrows)
    if log_scale:
        level_min = np.log2(nrows * ncols / Lx / Ly)
        level_max = int(np.log2(rho.max()))
        level_n = level_max - level_min + 1
        if level_n < 10:
            level_n = 2 * level_n - 1
        levels = np.linspace(level_min, level_max, level_n)
        c = ax.contourf(x, y, np.log2(rho), levels=levels, cmap=cmap)
        cb_label = r"$\log_2\left ({\rho}\right)$"
    else:
        rho[rho == 0] = None
        levels = np.linspace(0, rho_max, rho_max+1)
        c = ax.contourf(x, y, rho, levels=levels, cmap=cmap,
                        norm=PowerNorm(gamma=gamma), extend="max")
        cb_label = r"$\rho$"
    ax.axis("scaled")
    # ax.set_xlabel(r"$x$", fontsize="x-large")
    # ax.set_ylabel(r"$y$", fontsize="x-large")
    if flag_show:
        cb = plt.colorbar(c)
        cb.set_label(cb_label)
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return c


def plot_orient(vx, vy, Lx, Ly, num=None, ax=None):
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False
    nrows, ncols = vx.shape
    x = np.linspace(0.5 * Lx / ncols, Lx - 0.5 * Lx / ncols, ncols)
    y = np.linspace(0.5 * Ly / ncols, Ly - 0.5 * Ly / nrows, nrows)
    theta = np.arctan2(vy, vx) / np.pi * 180
    theta[theta < 0] += 360
    levels = np.linspace(0, 360, 13)
    if num is not None:
        theta[num == 0] = None
    c = ax.contourf(x, y, theta, levels=levels, cmap="hsv")
    ax.axis("scaled")
    # ax.set_xlabel(r"$x$", fontsize="x-large")
    # ax.set_ylabel(r"$y$", fontsize="x-large")
    if flag_show:
        cb = plt.colorbar(c)
        cb.set_label("orientation")
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return c


if __name__ == "__main__":
    os.chdir(r"D:\data\random_torque\phase diagram\L=1024\dump")
    sys.path.append(".")
    import decode
    file = r"cHff_0.25_0.02_1024_1024_512_512_1048576_111111.bin"
    frames = decode.decode_frames(file)
    for frame in frames:
        num, vx, vy, para = frame
        rho = num / 4
        print("std of rho =", np.std(rho), "max of rho is", rho.max())
        plot_density(rho, para["Lx"], para["Ly"], log_scale=True)
