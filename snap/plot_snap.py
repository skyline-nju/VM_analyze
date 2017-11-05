"""
    Plot coarse-grained snapshot of density and velocity fields.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
# import glob
import load_snap


def get_rgb(theta0, module0, scaled=True, module_max=None):
    theta, module = theta0.copy(), module0.copy()
    H = theta / 360
    if scaled:
        mmean, mstd = np.mean(module), np.std(module)
        mmax = min(module.max(), mmean + 2 * mstd)
        if module.min() < 0.05:
            mmin = 0
        else:
            mmin = max(module.min(), mmean - 2 * mstd)
        module[module > mmax] = mmax
        module[module < mmin] = mmin
        V = (module - mmin) / (mmax - mmin)
    else:
        V = module
        if module_max is not None:
            V[V > module_max] = module_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    if scaled:
        return RGB, mmin, mmax, mmean
    else:
        return RGB


def add_colorbar(ax, mmin, mmax, theta_min, theta_max):
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    t_min, t_max = 0, 360
    ax.imshow(
        RGB, origin='lower', extent=[t_min, t_max, mmin, mmax], aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
    ax.set_ylabel(r'module $\rho |v|$', fontsize="large")
    ax.set_xlabel("orientation", fontsize="large")


def plot(L, eta, eps, rho0, seed, t_end, rho, theta, module, polar_angle, phi,
         dl):
    rmax = rho.max()
    rmin = rho.min()
    rmean = np.mean(rho)
    rstd = np.std(rho)
    rho = (rho - rmean) / (2 * rstd)
    mmax = module.max()
    mmin = module.min()
    mstd = np.std(module)
    mmean = np.mean(module)
    # module2 = (module - mmean) / (2 * mstd)
    box = [0, L, 0, L]
    tick = np.arange(5) * L / 4
    x = y = np.linspace(dl / 2, L - dl / 2, L // dl)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 7))
    v = np.linspace(-1, 1, 11)
    cf = ax1.contourf(x, y, rho, v, cmap='hot', extend='both')
    ax1.axis('scaled')
    ax1.axis(box)

    ax1.set_xticks(tick)
    ax1.set_yticks(tick)

    RGB, mmin, mmax, mmean = get_rgb(theta, module)
    ax2.imshow(RGB, extent=[0, L, 0, L], origin='lower')
    ax2.axis('scaled')
    ax2.axis(box)
    ax2.set_xticks(tick)
    ax2.set_yticks(tick)

    plt.tight_layout()
    ax1.set_title(
        'density: min=%.3f, max=%.3f, mean=%.3f, SD=%.3f' % (rmin, rmax, rmean,
                                                             rstd),
        fontsize=11,
        color='b')
    ax2.set_title(
        'module: min=%.3f, max=%.3f, mean=%.3f, SD=%.3f' % (mmin, mmax, mmean,
                                                            mstd),
        fontsize=11,
        color='b')
    bbox1 = ax1.get_position().get_points().flatten()
    bbox2 = ax2.get_position().get_points().flatten()
    fig.subplots_adjust(bottom=0.2)
    bbox1[1], bbox1[3] = 0.14, 0.04
    bbox1[2] = bbox1[2] - bbox1[0] - 0.03
    bbox2[1], bbox2[3] = 0.08, 0.14
    bbox2[2] = bbox2[2] - bbox2[0]

    cb_ax1 = fig.add_axes(bbox1)
    cb_ax2 = fig.add_axes(bbox2)

    cb1 = fig.colorbar(cf, cax=cb_ax1, orientation='horizontal')
    cb1.set_label('density')
    cbticks = np.linspace(-1, 1, 6)
    cb1.set_ticks(cbticks)
    cb1.set_ticklabels(['%.4f' % i for i in cbticks * rstd * 2.0 + rmean])
    add_colorbar(cb_ax2, mmin, mmax, theta.min(), theta.max())

    # add_arrow
    length = L / 4
    ax1.arrow(
        L / 2,
        L / 2,
        length * np.cos(polar_angle),
        length * np.sin(polar_angle),
        color="k",
        width=3 * L / 2048)
    ax2.arrow(
        L / 2,
        L / 2,
        length * np.cos(polar_angle),
        length * np.sin(polar_angle),
        color="w",
        width=3 * L / 2048)

    fig.suptitle(
        r'$\eta=%.2f,\ \epsilon=%.3f,\ \langle \phi \rangle_t=%.4f,\ \rm{seed}=%d$'
        % (eta, eps, phi, seed),
        fontsize=18)

    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data")
    # file = r"cHff_0.1_0_8192_8192_1024_1024_67108864_17102532.bin"
    # file = r"cHff_0.18_0_8192_8192_1024_1024_67108864_17091901.bin"
    file = r"cHff_0.35_0_8192_8192_1024_1024_67108864_17092802.bin"
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
        # if t in [3200]:
        if t in [25, 50, 100, 200, 400, 800, 1600, 3200]:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
            rho = num / dA
            contour = ax1.contourf(x, y, rho, rho_level, extend="max")
            ax1.axis("scaled")
            ax1.axis(domain)

            v_orient = np.arctan2(vy, vx) / np.pi * 180
            v_orient[v_orient < 0] += 360
            v_module = np.sqrt(vx**2 + vy**2) * rho
            RGB = get_rgb(
                v_orient, v_module, scaled=False, module_max=4)
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
            # plt.show()
            plt.savefig("snap_%g_%g_%d_%04d.png" % (eta, eps, L, t))
            plt.close()
