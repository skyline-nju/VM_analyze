"""
Study ordering dynamics of the Vicsek model with and without the quenched
disorder.
"""
import numpy as np
import platform
import matplotlib
import os
# import sys
import glob
import load_snap
import spatial_corr as sc
from add_line import add_line

if platform.system() is not "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def plot_collapsed_curves(frames, L, eta, eps, dA, rho0=1):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, frame in enumerate(frames):
        t, vxm, vym, num, vx, vy = frame
        corr_rho, corr_v = sc.cal_corr2d(
            num, vx, vy, dA, vxm, vym, remove_mean=True)
        r, cv = sc.spherical_average(corr_v, L)
        r, crho = sc.spherical_average(corr_rho, L)
        Lc1 = sc.get_chara_length(r, cv, 0.5)
        Lc2 = sc.get_chara_length(r, crho, 0.5)
        line, = axes[0][0].plot(r, cv, label=r"$t=%d$" % t)
        axes[0][1].plot(r / Lc1, cv, color=line.get_color())
        axes[1][0].plot(r, crho, color=line.get_color())
        axes[1][1].plot(r / Lc2, crho, color=line.get_color())
    axes[0][0].legend()
    axes[0][0].set_xlabel(r"$r$")
    axes[0][0].set_ylabel(r"$C_V$")
    axes[0][1].set_xlabel(r"$r/L_V$")
    axes[0][1].set_ylabel(r"$C_V$")
    axes[1][0].set_xlabel(r"$r$")
    axes[1][0].set_ylabel(r"$C_{\rho}$")
    axes[1][1].set_xlabel(r"$r/L_{\rho}$")
    axes[1][1].set_ylabel(r"$C_{\rho}$")
    # axes[0][0].set_xlim(0, 300)
    # axes[1][0].set_xlim(0, 300)
    # axes[0][1].set_xlim(0, 15)
    # axes[1][1].set_xlim(0, 30)
    axes[0][0].set_ylim(ymax=1)
    axes[0][1].set_ylim(ymax=1)
    axes[1][0].set_ylim(ymax=1)
    axes[1][1].set_ylim(ymax=1)
    axes[0][0].axhline(y=0.5, linestyle="dashed", color="k")
    axes[0][1].axhline(y=0.5, linestyle="dashed", color="k")
    axes[1][0].axhline(y=0.5, linestyle="dashed", color="k")
    axes[1][1].axhline(y=0.5, linestyle="dashed", color="k")
    plt.suptitle(r"$L=%d,\ \eta=%g,\ \epsilon=%g,\ \rho_0=%g$" % (L, eta, eps,
                                                                  rho0))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def plot_lc(frames, L, eta, eps, dA, rho0=1, threshold=0.5):
    """ Show the growth of correlation length with increasing time. """
    t_array = []
    lc_rho = []
    lc_v = []
    for i, frame in enumerate(frames):
        t, vxm, vym, num, vx, vy = frame
        print("t=", t)
        corr_rho, corr_v = sc.cal_corr2d(
            num, vx, vy, dA, vxm, vym, remove_mean=True)
        r, cv = sc.spherical_average(corr_v, L)
        r, crho = sc.spherical_average(corr_rho, L)
        l1 = sc.get_chara_length(r, cv, threshold)
        l2 = sc.get_chara_length(r, crho, threshold)
        if l1 is not None and l2 is not None:
            t_array.append(t)
            lc_rho.append(l2)
            lc_v.append(l1)
    plt.loglog(t_array, lc_rho, "o", label=r"$C_{\rho}$")
    plt.loglog(t_array, lc_v, "o", label=r"$C_V$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$C(r)$")
    plt.legend()
    plt.show()
    plt.close()


def get_para(file):
    str_list = file.replace(".bin", "").split("_")
    L0 = int(str_list[3])
    eta = float(str_list[1])
    eps = float(str_list[2])
    ncols = int(str_list[5])
    nrows = int(str_list[6])
    seed = int(str_list[-1])
    return L0, eta, eps, ncols, nrows, seed


def sample_average(L0, eta, eps, ncols, frame_list=None):
    """ Calculate sample-averaged correlation function of density and velocity

    Parameters:
    --------
    L0 : int
        System size.
    eta : float
        Strength of noise.
    eps : float
        Strength of disorder.
    ncols : int
        Number of colunms of cells.
    frame_list : list
        Index of frame to read
    """
    if frame_list is None:
        if L0 == 2048:
            frame_list = [19, 26, 33, 41, 48, 55, 62, 70]
            pattern = "ciff_%g_%g_%d_%d_%d_%d_*_1.1_*.bin"
        else:
            frame_list = [0, 1, 3, 6, 12, 24, 48, 96]
            pattern = "ciff_%g_%g_%d_%d_%d_%d_*.bin"
        n_frame = len(frame_list)
    files = glob.glob(pattern % (eta, eps, L0, L0, ncols, ncols))
    print("%d files in total" % len(files))
    L0, eta, eps, ncols, nrows, seed = get_para(files[0])
    c_rho_sum = np.zeros((n_frame, nrows, ncols))
    c_v_sum = np.zeros((n_frame, nrows, ncols))
    cell_area = (L0 / ncols)**2
    t_list = []
    for i, file in enumerate(files):
        snap = load_snap.CoarseGrainSnap(file)
        frames = snap.gene_frames(frames=frame_list)
        for j, frame in enumerate(frames):
            t, vxm, vym, num, vx, vy = frame
            print("t=", t)
            c_rho, c_v = sc.cal_corr2d(
                num, vx, vy, cell_area, remove_mean=True)
            c_rho_sum[j] += c_rho
            c_v_sum[j] += c_v
            if i == 0:
                # print("t=", t)
                t_list.append(t)
    t_array = np.array(t_list)
    c_rho_mean = np.array([c / len(files) for c in c_rho_sum])
    c_v_mean = np.array([c / len(files) for c in c_v_sum])
    c_rho_r = []
    c_v_r = []
    for c_rho in c_rho_mean:
        r1, c1 = sc.spherical_average(c_rho, L0)
        c_rho_r.append(c1)
    for c_v in c_v_mean:
        r2, c2 = sc.spherical_average(c_v, L0)
        c_v_r.append(c2)
    outfile = "Crho_%d_%g_%g.npz" % (L0, eta, eps)
    np.savez(outfile, t=t_array, r=r1, c_rho_r=np.array(c_rho_r))
    outfile = "Cv_%d_%g_%g.npz" % (L0, eta, eps)
    np.savez(outfile, t=t_array, r=r2, c_v_r=np.array(c_v_r))


def sample_time_average(L0, eta, eps, ncols, n_frame=8):
    """ Each frame is averaged over samples and nearby time steps."""
    pattern = "ciff_%g_%g_%d_%d_%d_%d_*.bin"
    files = glob.glob(pattern % (eta, eps, L0, L0, ncols, ncols))
    L0, eta, eps, ncols, nrows, seed = get_para(files[0])
    c_rho_sum = np.zeros((n_frame, nrows, ncols))
    c_v_sum = np.zeros((n_frame, nrows, ncols))
    cell_area = (L0 / ncols)**2
    t_arr = np.array([25, 50, 99.5, 199.5, 399.5, 799.5, 1599.5, 3199.5])
    count = np.zeros(n_frame, int)
    for i, file in enumerate(files):
        snap = load_snap.CoarseGrainSnap(file)
        frames = snap.gene_frames()
        k = 0
        t_pre = 0
        for frame in frames:
            t, vxm, vym, num, vx, vy = frame
            c_rho, c_v = sc.cal_corr2d(
                num, vx, vy, cell_area, remove_mean=True)
            if t_pre == 0:
                t_pre = t
            if t - t_pre > 1:
                k += 1
            t_pre = t
            c_rho_sum[k] += c_rho
            c_v_sum[k] += c_v
            count[k] += 1
    for i in range(n_frame):
        c_rho_sum[i] /= count[i]
        c_v_sum[i] /= count[i]
    c_rho_r = []
    c_v_r = []
    for c_rho in c_rho_sum:
        r1, c1 = sc.spherical_average(c_rho, L0)
        c_rho_r.append(c1)
    for c_v in c_v_sum:
        r2, c2 = sc.spherical_average(c_v, L0)
        c_v_r.append(c2)
    outfile = "Crhot_%d_%g_%g.npz" % (L0, eta, eps)
    np.savez(outfile, t=t_arr, r=r1, c_rho_r=np.array(c_rho_r))
    outfile = "Cvt_%d_%g_%g.npz" % (L0, eta, eps)
    np.savez(outfile, t=t_arr, r=r2, c_v_r=np.array(c_v_r))


def plot_sample_averaged_Crho(L0, eta, eps):
    """
    Plot sample-averaged correlation function of density.

    The obtained correlation functions decay as a power law with a exponential
    cutoff, which increases with increasing time.

    It still need to be improved how to locate positions of exponential cutoff.
    """
    file = "Crho_%d_%g_%g.npz" % (L0, eta, eps)
    data = np.load(file)
    t = data["t"]
    r = data["r"]
    c_rho_r = data["c_rho_r"]
    if eps == 0.02:
        rcut = [8.14, 16.4, 33.3, 64, 124, 242, 370, 458]
    elif eps == 0.04:
        rcut = [8.3, 14.88, 30.57, 53.7, 104, 208, 417, 339]
    elif eps == 0.06:
        rcut = [7.2, 13.5, 25, 40.7, 76.9, 127.5, 134.7, 159]
    else:
        rcut = [8.21, 15.25, 33.3, 60, 122.8, 233.1, 350, 410]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for i, cr in enumerate(c_rho_r):
        if i == 2 or i == 3 or i == 4 or i == 5:
            ax1.plot(r, cr, "o", label=r"$%d$" % (t[i]), ms=2)
            ax2.plot(r / rcut[i], cr, "o", label=r"$%d$" % (t[i]), ms=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$C_{\rho}(r)$")
    ax1.legend(loc="upper right", title=r"$t=$", fontsize="x-small")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlim(xmax=2)
    ax2.set_xlabel(r"$r/\xi$")
    ax2.set_ylabel(r"$C_{\rho}(r)$")
    ax2.legend(loc="lower left", title=r"$t=$", fontsize="x-small")

    ax3.loglog(t, rcut, "o")
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\xi$")
    add_line(ax3, 0, 0.1, 1, 1, scale="log", label="slope 1", xl=0.2, yl=0.8)

    plt.suptitle(r"$L=%d, \eta=%g, \epsilon=%g, \rho_0=1$" % (L0, eta, eps))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def plot_sample_averaged_Cv(L0, eta, eps):
    """
    Plot sample-averaged correlation function of velocity.

    The obtained correlation functions decay as a power law with a exponential
    cutoff, which increases with increasing time.

    It still need to be improved how to locate positions of exponential cutoff.
    """
    file = "Cv_%d_%g_%g.npz" % (L0, eta, eps)
    data = np.load(file)
    t = data["t"]
    r = data["r"]
    c_rho_r = data["c_v_r"]
    if eps == 0.02:
        rcut = [8.14, 16.4, 33.3, 64, 124, 242, 370, 458]
    elif eps == 0.04:
        rcut = [8.3, 14.88, 30.57, 53.7, 104, 208, 417, 339]
    elif eps == 0.06:
        rcut = [7.2, 13.5, 25, 40.7, 76.9, 127.5, 134.7, 159]
    else:
        rcut = [8.21, 15.25, 33.3, 60, 122.8, 233.1, 350, 410]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for i, cr in enumerate(c_rho_r):
        ax1.plot(r, cr, label=r"$%d$" % (t[i]), ms=2)
        ax2.plot(r / rcut[i], cr, "o", label=r"$%d$" % (t[i]), ms=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$C_{\rho}(r)$")
    ax1.legend(loc="upper right", title=r"$t=$", fontsize="x-small")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlim(xmax=2)
    ax2.set_xlabel(r"$r/\xi$")
    ax2.set_ylabel(r"$C_{\rho}(r)$")
    ax2.legend(loc="lower left", title=r"$t=$", fontsize="x-small")

    ax3.loglog(t, rcut, "o")
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\xi$")
    add_line(ax3, 0, 0.1, 1, 1, scale="log", label="slope 1", xl=0.2, yl=0.8)

    plt.suptitle(r"$L=%d, \eta=%g, \epsilon=%g, \rho_0=1$" % (L0, eta, eps))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def show_comparision(L0, eta, eps):
    file1 = "Crho_%d_%g_%g.npz" % (L0, eta, eps)
    file2 = "Crhot_%d_%g_%g.npz" % (L0, eta, eps)
    data1 = np.load(file1)
    t1 = data1["t"]
    r1 = data1["r"]
    c_rho_r1 = data1["c_rho_r"]
    data2 = np.load(file2)
    t2 = data2["t"]
    r2 = data2["r"]
    c_rho_r2 = data2["c_rho_r"]

    plt.plot(r1, c_rho_r1[-2], "o", ms=1, label=r"$t=3200$")
    plt.plot(
        r2, c_rho_r2[-2], "s", ms=1, label=r"$t\in \left[3168, 3231 \right]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    for i, cr in enumerate(c_rho_r1):
        ax1.plot(r1, cr, label=r"$%d$" % (t1[i]))
    for i, cr in enumerate(c_rho_r2):
        ax2.plot(r2, cr, label=r"$%g$" % (t2[i]))

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 1)
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$C_{\rho}(r)$")
    ax1.legend(loc="upper right", title=r"$t=$", fontsize="x-small")
    ax1.set_title("instantaneous")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$C_{\rho}(r)$")
    ax2.legend(loc="upper right", title=r"$t=$", fontsize="x-small")
    ax2.set_title("with a bit of time average" +
                  r" $\Delta t = 1, 1, 2, 4, 8, 16, 32, 64$")

    plt.suptitle(r"$L=%d, \eta=%g, \epsilon=%g, \rho_0=1$, box size = 4" %
                 (L0, eta, eps))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


if __name__ == "__main__":
    if platform.system() is "Windows":
        # os.chdir(r"D:\code\corr2d\data")
        # file = r"ciff_0.18_0_2048_2048_1024_1024_4194304_1.06_123.bin"
        # file = r"ciff_0.18_0_4096_4096_1024_1024_16777216_1.06_123.bin"
        os.chdir(r"E:\data\random_torque\ordering")
        # os.chdir(r"D:\data\statistic")
        # file = r"cHff_0.18_0.08_1024_1024_1024_1024_1048576_1709040.bin"
    else:
        os.chdir(r"coarse")

    file = r"ciff_0.18_0_2048_2048_1024_1024_4194304_1.1_17090301.bin"

    L0, eta, eps, ncols, nrows, seed = get_para(file)
    cell_area = (L0 / ncols)**2
    snap = load_snap.CoarseGrainSnap(file)
    print(snap.get_tot_frames_num())
    frames = snap.gene_frames()
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t == 26:
            c_rho, c_v = sc.cal_corr2d(
                num, vx, vy, cell_area, remove_mean=True)
            sc.vary_box_size([2], L0, t, num, vx, vy, rm_mean=True)

    # plot_sample_averaged_Crho(2048, 0.18, 0)
    # sample_average(2048, 0.18, 0.02, 1024)
    # sample_time_average(4096, 0.18, 0, 1024)
    # show_comparision(4096, 0.18, 0)
