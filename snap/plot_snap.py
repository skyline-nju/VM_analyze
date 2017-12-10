"""
    Plot coarse-grained snapshot of density and velocity fields.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
# import glob
import load_snap


def get_rgb(theta0, module0, m_max=None):
    """ Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta0: array_like
        Orietation of velocity field.
    module0: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    theta, module = theta0.copy(), module0.copy()
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $\rho |v|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        ax.set_xlabel(r"module $\rho |v|$", fontsize="large")


def plot_two_panel(file, t_list=None, save=False):
    """ Plot density and velocity filed on left, right pannel, respectively.
        One picture per frame.

    Parameters:
    ---------
    file: str
        Input file
    t_list: array_like, optional
        The frames to show. If is None, t_list = [25, 50, 100, 200, 400, 800,
        1600, 3200]
    save: bool, optional
        Whether to save the figure to disk.
    """
    if t_list is None:
        t_list = [25, 50, 100, 200, 400, 800, 1600, 3200]
    snap = load_snap.CoarseGrainSnap(file)
    frames = snap.gene_frames()
    s = file.replace(".bin", "").split("_")
    eta = float(s[1])
    eps = float(s[2])
    L = int(s[3])
    ncols = int(s[5])
    domain = [0, L, 0, L]
    lBox = L // ncols
    dA = lBox**2
    x = y = np.linspace(lBox / 2, L - lBox / 2, ncols)
    # rho_level = np.linspace(0, 5, 11)
    rho_level = np.linspace(0, 4, 9)
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t in t_list:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
            rho = num / dA
            contour = ax1.contourf(x, y, rho, rho_level, extend="max")
            ax1.axis("scaled")
            ax1.axis(domain)

            v_orient = np.arctan2(vy, vx) / np.pi * 180
            v_orient[v_orient < 0] += 360
            v_module = np.sqrt(vx**2 + vy**2) * rho
            RGB = get_rgb(v_orient, v_module, m_max=4)
            ax2.imshow(RGB, extent=[0, L, 0, L], origin="lower")
            ax2.axis('scaled')
            ax2.axis(domain)
            plt.suptitle(
                r"$\eta=%g,\ \rho_0=1, \epsilon=%g,\ L=%d,\ t=%d$" % (eta, eps,
                                                                      L, t),
                fontsize="xx-large",
                y=0.985)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            bbox1 = ax1.get_position().get_points().flatten()
            bbox2 = ax2.get_position().get_points().flatten()
            fig.subplots_adjust(bottom=0.24)
            bbox1[1], bbox1[3] = 0.14, 0.04
            bbox1[2] = bbox1[2] - bbox1[0] - 0.03
            bbox2[1], bbox2[3] = 0.08, 0.14
            bbox2[2] = bbox2[2] - bbox2[0]
            cb_ax1 = fig.add_axes(bbox1)
            cb_ax2 = fig.add_axes(bbox2)
            cb1 = fig.colorbar(contour, cax=cb_ax1, orientation="horizontal")
            cb1.set_label(r"density $\rho$", fontsize="x-large")
            add_colorbar(cb_ax2, v_module.min(), 4, 0, 360)
            if save:
                plt.savefig("snap_%g_%g_%d_%04d.jpg" % (eta * 100, eps, L, t))
            else:
                plt.show()
            plt.close()


def plot_serial_snap(file, save=False, rescale=False):
    """ Plot density (upper row) and velocity (lower row), with incresing time
        from left to right.

    Parameters:
    --------
    file: str
        Input file.
    save: bool, optional
        If true, save the figure into disk.
    rescale: bool, optional
        If true, xlim, ylim increase linearly with incresing time.
    """
    t_list = [400, 800, 1600, 3200]
    snap = load_snap.CoarseGrainSnap(file)
    frames = snap.gene_frames()
    s = file.replace(".bin", "").split("_")
    eta = float(s[1])
    eps = float(s[2])
    L = int(s[3])
    ncols = int(s[5])
    lBox = L // ncols
    dA = lBox**2
    x = y = np.linspace(lBox / 2, L - lBox / 2, ncols)
    rho_level = np.linspace(0, 3, 7)
    fig, axes = plt.subplots(nrows=2, ncols=len(t_list), figsize=(12, 6))
    col = 0
    if rescale:
        i = ncols // 8
    else:
        i = ncols
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t in t_list:
            rho = num / dA
            contour = axes[0][col].contourf(
                x[0:i], y[0:i], rho[0:i, 0:i], rho_level, extend="max")
            axes[0][col].axis("scaled")
            axes[0][col].axis([0, i * lBox, 0, i * lBox])
            axes[0][col].axis("off")
            axes[0][col].set_title(r"$t=%d$" % (t), fontsize="x-large")

            v_orient = np.arctan2(vy[0:i, 0:i], vx[0:i, 0:i]) / np.pi * 180
            v_orient[v_orient < 0] += 360
            v_module = np.sqrt(vx[0:i, 0:i]**2 + vy[0:i, 0:i]**2) * rho[0:i, 0:
                                                                        i]
            RGB = get_rgb(v_orient, v_module, m_max=4)
            axes[1][col].axis('scaled')
            axes[1][col].imshow(
                RGB, extent=[0, i * lBox, 0, i * lBox], origin="lower")
            axes[1][col].axis([0, i * lBox, 0, i * lBox])
            axes[1][col].axis("off")
            col += 1
            if rescale:
                i *= 2

    # axes[0][0].set_title("density", fontsize="x-large", loc="left")
    # axes[1][0].set_title("velocity", fontsize="x-large", loc="left")
    plt.suptitle(
        r"$\eta=%g,\ \rho_0=1,\ \epsilon=%g,\ L=%d$" % (eta, eps, L),
        fontsize="xx-large",
        y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.005, w_pad=0.005)

    fig.subplots_adjust(right=0.90)

    # positon of the last column: x0, y0, x1, y1
    bbox1 = axes[0][col - 1].get_position().get_points().flatten()
    bbox2 = axes[1][col - 1].get_position().get_points().flatten()

    # add axes for colobar, whose position is [left, bottom, width, height]
    dy = 0.015
    cb_ax1 = fig.add_axes(
        [0.92, bbox1[1] + dy, 0.02, bbox1[3] - bbox1[1] - 2 * dy])
    cb1 = fig.colorbar(contour, cax=cb_ax1)
    cb1.set_label(r"density $\rho$", fontsize="x-large")
    cb_ax2 = fig.add_axes(
        [0.91, bbox2[1] + dy, 0.04, bbox2[3] - bbox2[1] - 2 * dy])
    add_colorbar(cb_ax2, 0, 4, 0, 360, orientation="v")
    if save:
        plt.savefig(
            r"../fig/snap_%d_%g_%g.jpg" % (L, eta * 100, eps * 100),
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data")
    # file = r"cHff_0.1_0_8192_8192_1024_1024_67108864_17102532.bin"
    # file = r"cHff_0.18_0_8192_8192_1024_1024_67108864_17091901.bin"
    # file = r"cHff_0.18_0_8192_8192_4096_4096_67108864_17111451.bin"
    file = r"cHff_0.35_0_8192_8192_1024_1024_67108864_17092802.bin"
    # file = r"cHff_0.18_0.02_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.18_0.04_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.18_0.06_8192_8192_1024_1024_67108864_17120201.bin"
    # file = r"cHff_0.4_0_8192_8192_1024_1024_67108864_17110541.bin"
    # plot_two_panel(file)
    plot_serial_snap(file, save=True, rescale=False)
