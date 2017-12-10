""" Load the binary file where the spherically averaged correlation functions
    and structure factor for density and velocity are saved.
"""

import struct
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from add_line import add_line

# from spatial_corr import get_chara_length


def read(file, load_Sk=False):
    """ Read a file contaning Cr (and Sk).

        File name looks like cr_%g_%g_%g_%d_%d_*.bin or
        crsk_%g_%g_%g_%d_%d_*.bin.
    """
    s = file.replace(".bin", "").split("_")
    size_r = int(s[-2])
    tot_frames = int(s[-1])
    cell_area = int(s[5])**2
    t = np.zeros(tot_frames, int)
    rho_m = np.zeros(tot_frames)
    vx_m = np.zeros_like(rho_m)
    vy_m = np.zeros_like(rho_m)
    crho_r = np.zeros((tot_frames, size_r))
    cv_r = np.zeros_like(crho_r)
    if load_Sk:
        srho_k = np.zeros_like(crho_r)
        sv_k = np.zeros_like(crho_r)
    if "sk" in file:
        flag_Sk = True
    else:
        flag_Sk = False

    with open(file, "rb") as f:
        buff = f.read(8 * size_r)
        r = np.array(struct.unpack("%dd" % size_r, buff))
        print(file)
        for i in range(tot_frames):
            buff = f.read(4)
            t[i], = struct.unpack("i", buff)
            buff = f.read(24)
            rho_m[i], vx_m[i], vy_m[i] = struct.unpack("ddd", buff)
            vx_m[i] /= cell_area
            vy_m[i] /= cell_area
            # print(t[i], vx_m[i], vy_m[i], np.sqrt(vx_m[i]**2 + vy_m[i]**2))
            buff = f.read(8 * size_r)
            crho_r[i] = np.array(struct.unpack("%dd" % size_r, buff))
            buff = f.read(8 * size_r)
            cv_r[i] = np.array(struct.unpack("%dd" % size_r, buff))
            if load_Sk:
                if not flag_Sk:
                    print("Fail to load Sk in %s" % file)
                    sys.exit()
                else:
                    srho_k[i] = np.array(struct.unpack("%dd" % size_r, buff))
                    sv_k[i] = np.array(struct.unpack("%dd" % size_r, buff))
            elif flag_Sk:
                f.seek(size_r * 16, 1)
    if load_Sk:
        return t, rho_m, vx_m, vy_m, r, crho_r, cv_r, srho_k, sv_k
    else:
        return t, rho_m, vx_m, vy_m, r, crho_r, cv_r


def split_array(a, sep=1):
    """
    Split input array `a` into several serials, so that for each serial, the
    difference between two nearest numbers is not more than `sep`.
    """
    pos = [0]
    a_pre = a[0]
    for i, a_cur in enumerate(a):
        if a_cur > a_pre + sep:
            pos.append(i)
        a_pre = a_cur
    pos.append(i + 1)
    return pos


def plot_lin_fit(xdata, ydata, ax, scale="lin"):
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 1000)
    if scale == "lin":
        p = np.polyfit(xdata, ydata, 1)
        y = p[0] * x + p[1]
    else:
        p = np.polyfit(np.log10(xdata), np.log10(ydata), 1)
        y = 10**(p[0] * np.log10(x) + p[1])
    ax.plot(x, y, "--", label="slope = %.3f" % (p[0]))


def time_ave(t, rho_m, phi, crho_r, cv_r, srho_k=None, sv_k=None):
    """ Average over nearby frames. """
    pos = split_array(t)
    n = len(pos) - 1
    t_new = np.zeros(n)
    rho_m_new = np.zeros(n)
    phi_new = np.zeros(n)
    crho_r_new = np.zeros((n, crho_r.shape[1]))
    cv_r_new = np.zeros_like(crho_r_new)
    if srho_k is not None:
        srho_k_new = np.zeros_like(crho_r_new)
        sv_k_new = np.zeros_like(crho_r_new)
    for i in range(n):
        j, k = pos[i], pos[i + 1]
        t_new[i] = np.mean(t[j:k])
        phi_new[i] = np.mean(phi[j:k])
        crho_r_new[i] = np.mean(crho_r[j:k], axis=0)
        cv_r_new[i] = np.mean(cv_r[j:k], axis=0)
        if srho_k is not None:
            srho_k_new[i] = np.mean(srho_k[j:k], axis=0)
            sv_k_new[i] = np.mean(sv_k[j:k], axis=0)
    if srho_k is None:
        return t_new, rho_m_new, phi_new, crho_r_new, cv_r_new
    else:
        return t_new, rho_m_new, phi_new, crho_r_new, cv_r_new, \
            srho_k_new, sv_k_new


def sample_ave(eta, eps, L, l, rho0=1, t_ave=True, files=None):
    """ Average over samples.

    Parameters:
    --------
    eta: float
        Noise strength.
    eps: float
        Disorder strength.
    L: int
        System size.
    rho0: float, optional
        Particle density
    t_ave: bool, optional
        Whether do average over time.
    files: list of str, optional
        Input files.

    Returns:
    --------
    t: array_like
        Time for each frames.
    rho_m: array_like
        Mean density.
    phi: array_like
        Order parameters.
    r: array_like
        Array of distance from the origin.
    crho_r: array_like
        Spherically averaged correlation function of density.
    cv_r: array_like
        Spherically averaged correlation function of velocity.
    """
    if files is None:
        files = glob.glob("cr*_%g_%g_%g_%d_%d_*.bin" % (eta, eps, rho0, L, l))
    t, rho_m, vx_m, vy_m, r, crho_r, cv_r = read(files[0])
    phi = np.sqrt(vx_m**2 + vy_m**2)
    for file in files[1:]:
        print(file)
        t, rho_m2, vx_m2, vy_m2, r, crho_r2, cv_r2 = read(file)
        rho_m += rho_m2
        phi += np.sqrt(vx_m2**2 + vy_m2**2)
        crho_r += crho_r2
        cv_r += cv_r2
    if len(files) > 1:
        rho_m /= len(files)
        phi /= len(files)
        crho_r /= len(files)
        cv_r /= len(files)
    if t_ave:
        t, rho_m, phi, crho_r, cv_r = time_ave(t, rho_m, phi, crho_r, cv_r)
    return t, rho_m, phi, r, crho_r, cv_r


def plot_cr(t,
            r,
            cr,
            start_idx=0,
            lc=None,
            mean=0,
            xlog=True,
            ylog=True,
            t_specify=None,
            threshold=None,
            marker="o",
            ms=2,
            clist=[],
            flag_line_label=True,
            normed=True,
            ax=None):
    """ Plot correlatin function vs. distance. """
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    else:
        flag_show = False

    if len(clist) == 0:
        flag_c = True
    else:
        flag_c = False
    cr = cr[start_idx:]
    t = t[start_idx:]
    if isinstance(mean, np.ndarray):
        mean = mean[start_idx:]
    for i, cr_t in enumerate(cr):
        if t_specify is None or t[i] in t_specify:
            if lc is None:
                r_new = r
            else:
                r_new = r / lc[start_idx + i]
            if isinstance(mean, np.ndarray):
                c_new = cr_t - mean[i]
                if normed:
                    c_new /= (cr_t[0] - mean[i])
            else:
                c_new = cr_t - mean
                if normed:
                    c_new /= (cr_t[0] - mean)
            line, = ax.plot(r_new, c_new, marker, ms=ms)
            if threshold is not None:
                xi = get_crossing_point(r_new, c_new, threshold, scale="log")
                # ax.axvline(xi)
                print(t[i], xi)
            if flag_c:
                clist.append(line.get_color())
            else:
                line.set_color(clist[i])
            if flag_line_label:
                if t[i] > 25:
                    line.set_label("%g" % (t[i] + 0.5))
                else:
                    line.set_label("25")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if flag_show:
        ax.legend(title=r"$t=$")
        plt.show()
        plt.close()


def varied_eta(*args,
               L=8192,
               eps=0,
               rho0=1,
               l=2,
               start_idx=0,
               save=False,
               normed=False,
               hline=None):
    """ Plot correlation functions with varied eta and incresing time.

    Parameters:
    --------
    *args: float
        Values of eta.
    L: int, optional
        System size.
    eps: float, optional
        Strength of quenched disorder.
    rho0: float, optional
        Particle density.
    l: float, optional
        Boxes size for coarse grain.
    start_idx: int, optional
        The first `start_idx` points are not shown.
    save: bool, optional
        Whether to save figure.
    normed: bool, optional,
        If `True`, normalize the correlation function with it's value at r=0.
    hline: float, optional
        Plot a horizontal line at `y=hline` on the right panel.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    mk = {0.10: "-", 0.18: "o", 0.35: "--", 0.4: "-."}
    line_type = {
        0.10: "solid line",
        0.18: "circle",
        0.35: "dashed line",
        0.4: "dash dotted"
    }
    title = ""
    clist = []
    for i, eta in enumerate(args):
        t, rho_m, phi, r, crho_r, cv_r = sample_ave(eta, eps, L, l)
        print("phi= ", phi)
        if i == 0:
            flag_line_label = True
        else:
            flag_line_label = False
        plot_cr(
            t,
            r,
            crho_r,
            start_idx=start_idx,
            mean=1,
            ax=ax1,
            marker=mk[eta],
            normed=normed,
            xlog=True,
            ylog=True,
            flag_line_label=flag_line_label,
            clist=clist)
        if normed is False:
            mean = 0
        else:
            mean = phi**2
        plot_cr(
            t,
            r,
            cv_r,
            xlog=True,
            ylog=True,
            start_idx=start_idx,
            ax=ax2,
            mean=mean,
            marker=mk[eta],
            flag_line_label=flag_line_label,
            threshold=hline,
            normed=normed,
            clist=clist)
        title += " %g (%s)," % (eta, line_type[eta])
    title = title[:-1]  # remove the last comma
    ax1.legend(title=r"$t=$", fontsize="large")
    ax2.legend(title=r"$t=$", fontsize="large")
    if normed:
        ylabel1 = r"$C_\rho (r, t)/C_\rho (0, t)$"
        ylabel2 = r"$C_v (r, t) / C_v (0, t)$"
    else:
        ylabel1 = r"$C_\rho (r, t)$"
        ylabel2 = r"$C_v (r, t)$"
    ax1.set_xlabel(r"$r$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(ylabel1, fontsize="x-large")
    ax2.set_ylabel(ylabel2, fontsize="x-large")
    # ax1.set_xlim(xmax=2e3)
    # ax2.set_xlim(xmax=5e3)
    # ax1.set_ylim(1e-4)
    # ax2.set_ylim(1e-4)
    # if normed:
    #     ax1.set_ylim(5e-5, 1)
    # else:
    #     if 0.1 in args:
    #         ax1.set_ylim(1e-4, 10)
    #     else:
    #         ax1.set_ylim(1e-5, 3)
    # ax2.set_ylim(1e-5, 1)
    if hline is not None:
        ax2.axhline(hline, color="k")

    title = r"$L=%d,\ \rho_0=%g,\ \epsilon=%g,\ \eta=$" + title
    plt.suptitle(title % (L, rho0, eps), fontsize="xx-large", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save:
        filename = "eta"
        for eta in args:
            filename += "_%g" % eta
        plt.savefig(filename + ".pdf")
    else:
        plt.show()
    plt.close()


def varied_L(*args,
             eta=0.18,
             eps=0,
             rho0=1,
             l=2,
             start_idx=0,
             normed=False,
             save=False):
    """ Plot correlation functions with varied system size and time.

    Parameters:
    --------
    *args: float
        Values of system size.
    eta: float, optional
        Strength of noise.
    eps: float, optional
        Strength of quenched disorder.
    rho0: float, optional
        Particle density.
    l: float, optional
        Boxes size for coarse grain.
    start_idx: int, optional
        The first `start_idx` points are not shown.
    normed: bool, optional,
        If `True`, normalize the correlation function with it's value at r=0.
    save: bool, optional
        Whether to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    mk = {2048: ":", 4096: "--", 8192: "s"}
    line_type = {2048: "dotted line", 4096: "dashed line", 8192: "square"}
    title = ""
    clist = []
    for i, L in enumerate(args):
        t, rho_m, phi, r, crho_r, cv_r = sample_ave(eta, eps, L, l)
        if i == 0:
            flag_line_label = True
        else:
            flag_line_label = False
        plot_cr(
            t,
            r,
            crho_r,
            start_idx,
            mean=1,
            ax=ax1,
            marker=mk[L],
            ms=3,
            flag_line_label=flag_line_label,
            normed=normed,
            clist=clist)
        plot_cr(
            t,
            r,
            cv_r,
            xlog=True,
            start_idx=start_idx,
            ax=ax2,
            marker=mk[L],
            ms=3,
            flag_line_label=flag_line_label,
            normed=normed,
            clist=clist)
        title += " %g (%s)," % (L, line_type[L])
    title = title[:-1]  # remove the last comma
    ax1.legend(title=r"$t=$", fontsize="large")
    ax2.legend(title=r"$t=$", fontsize="large")
    if normed:
        ylabel1 = r"$C_\rho (r, t)/C_\rho (0, t)$"
        ylabel2 = r"$C_v (r, t) / C_v (0, t)$"
    else:
        ylabel1 = r"$C_\rho (r, t)$"
        ylabel2 = r"$C_v (r, t)$"
    ax1.set_xlabel(r"$r$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(ylabel1, fontsize="x-large")
    ax2.set_ylabel(ylabel2, fontsize="x-large")
    if normed:
        ax1.set_ylim(5e-5, 1)
    else:
        ax1.set_ylim(1e-4, 3)
    ax2.set_ylim(1e-3, 1)
    ax1.set_xlim(xmax=2e3)
    ax2.set_xlim(xmax=4e3)

    title = r"$\eta=%g,\ \rho_0=%g,\ \epsilon=%g,\ L=$" + title
    plt.suptitle(title % (eta, rho0, eps), fontsize="xx-large", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save:
        filename = "L"
        for L in args:
            filename += "_%g" % L
        plt.savefig(filename + ".eps")
    else:
        plt.show()
    plt.close()


def collapse6(eta,
              eps=0,
              rho0=1,
              L=8192,
              l=2,
              beg_idx=2,
              normed=False,
              save=False):
    """ Show the collapse of rescaled correlation functions. Two rows and
        three coloums, total 6 pannels.

    Parameters:
    --------
    eta: float
        Strength of noise.
    eps: float, optional
        Stregth of disorder.
    rho0: float, optional
        Particle density.
    L: int, optional
        System size.
    l: int, optional
        Boxes size for coarse grain.
    beg_idx: int, optional
        The first `beg_idx` points are not shown.
    normed: bool, optional,
        If `True`, normalize the correlation function with it's value at r=0.
    save: bool, optional
        Whether to save figure.
    """
    t, rho_m, phi, r, crho_r, cv_r = sample_ave(eta, eps, L, l)
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
    if eta == 0.18:
        lc_v = dict_lc["v"][eta][eps]
    else:
        lc_v = dict_lc["v"][eta]
    # lc_rho = dict_lc["rho"][eta]
    lc_rho = lc_v
    t, rho_m, phi, r, crho_r, cv_r = sample_ave(eta, eps, L, l)
    label_size = "x-large"
    if normed:
        ylabel1 = r"$C_\rho (r, t)/C_\rho (0, t)$"
        ylabel2 = r"$C_v (r, t) / C_v (0, t)$"
    else:
        ylabel1 = r"$C_\rho (r, t)$"
        ylabel2 = r"$C_v (r, t)$"
    """ density """
    plot_cr(t, r, crho_r, beg_idx, mean=1, ax=axes[0][0], normed=normed)
    plot_cr(
        t, r, crho_r, beg_idx, lc_rho, mean=1, ax=axes[0][1], normed=normed)
    plot_cr(
        t,
        r,
        crho_r,
        beg_idx,
        lc_rho,
        mean=1,
        xlog=False,
        ylog=False,
        normed=normed,
        ax=axes[0][2])
    axes[0][0].set_xlabel(r"$r$", fontsize=label_size)
    axes[0][0].set_ylabel(ylabel1, fontsize=label_size)
    axes[0][1].set_xlabel(r"$r/\xi_v$", fontsize=label_size)
    axes[0][1].set_ylabel(ylabel1, fontsize=label_size)
    axes[0][2].set_xlabel(r"$r/\xi_v$", fontsize=label_size)
    axes[0][2].set_ylabel(ylabel1, fontsize=label_size)
    axes[0][1].legend(title="t=")
    axes[0][2].legend(title="t=")
    axes[0][1].set_xlim(xmax=1)
    axes[0][2].set_xlim(-0.02, 0.5)
    if normed:
        axes[0][0].set_ylim(1e-4, 1)
        axes[0][1].set_ylim(1e-4, 1)
        axes[0][2].set_ylim(-0.05, 1)
    else:
        axes[0][0].set_ylim(1e-4)
        axes[0][1].set_ylim(1e-4)
        axes[0][2].set_ylim(-0.3)
    """ velocity """
    if normed is False:
        mean = 0
    else:
        mean = phi**2
    plot_cr(t, r, cv_r, beg_idx, mean=mean, ax=axes[1][0], normed=normed)
    plot_cr(t, r, cv_r, beg_idx, lc_v, mean=mean, ax=axes[1][1], normed=normed)
    plot_cr(
        t,
        r,
        cv_r,
        beg_idx,
        lc_v,
        mean=mean,
        xlog=False,
        ylog=False,
        ax=axes[1][2],
        normed=normed)
    axes[1][1].set_xlabel(r"$r/\xi_v$", fontsize=label_size)
    axes[1][1].set_ylabel(ylabel2, fontsize=label_size)
    axes[1][0].set_xlabel(r"$r$", fontsize=label_size)
    axes[1][0].set_ylabel(ylabel2, fontsize=label_size)
    axes[1][2].set_xlabel(r"$r/\xi_v$", fontsize=label_size)
    axes[1][2].set_ylabel(ylabel2, fontsize=label_size)
    axes[1][1].legend(title="t=")
    axes[1][2].legend(title="t=")
    axes[1][0].set_ylim(1e-3, 1)
    axes[1][1].set_xlim(xmax=11)
    axes[1][1].set_ylim(1e-3, 1)
    axes[1][2].set_xlim(-0.02, 0.5)
    axes[1][2].set_ylim(-0.05, 1)
    plt.suptitle(
        r"$L=%d,\ \eta=%g, \rho_0=%d,\ \epsilon=%g$" % (L, eta, rho0, eps),
        fontsize="xx-large",
        y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save:
        plt.savefig("collapse_%d_%g_%d.png" % (L, eta, beg_idx))
    else:
        plt.show()
    plt.close()


def plot_lc(save=False, L=8192):
    """ Plot correlation length as a function of time in log-log scales."""
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    p = np.polyfit(np.log10(t_specify[2:]), np.log10(dict_lc["v"][0.1][2:]), 1)
    plt.plot(
        t_specify,
        dict_lc["v"][0.1],
        "o",
        label="$\eta=%g$, slope=%.3f" % (0.1, p[0]))
    p = np.polyfit(
        np.log10(t_specify[2:]), np.log10(dict_lc["v"][0.18][0][2:]), 1)
    plt.plot(
        t_specify,
        dict_lc["v"][0.18][0],
        "s",
        label="$\eta=%g$, slope=%.3f" % (0.18, p[0]))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize="medium")
    plt.xlabel(r"$t$", fontsize="xx-large")
    plt.ylabel(r"$\xi_v$", fontsize="xx-large")
    add_line(plt.gca(), 0, 0.08, 1, 1, scale="log", label="slope 1")
    add_line(plt.gca(), 0.1, 0, 1, 0.95, "slope 0.95", 0.6, 0.5, scale="log")
    plt.subplot(122)
    plt.plot(t_specify, dict_lc["v"][0.1] / t_specify, "o")
    plt.plot(t_specify, dict_lc["v"][0.18][0] / t_specify, "s")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$t$", fontsize="xx-large")
    plt.ylabel(r"$\xi_v/t$", fontsize="xx-large")
    add_line(plt.gca(), 0, 0.8, 1, -0.05, "slope -0.05", scale="log")

    plt.suptitle(
        r"$L=%d,\ \rho_0=%g,\ \epsilon=%g$" % (L, 1, 0), fontsize="xx-large")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig("Lc_%d.pdf" % (L))
    else:
        plt.show()
    plt.close()


def plot_lc2(save=False, L=8192):
    """ Plot correlation length as a function of time in log-log scales."""
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    for eps in [0, 0.02, 0.04]:
        plt.plot(
            t_specify,
            dict_lc["v"][0.18][eps],
            "o",
            label=r"$\epsilon=%g$" % eps)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize="medium")
    plt.xlabel(r"$t$", fontsize="xx-large")
    plt.ylabel(r"$\xi_v$", fontsize="xx-large")
    add_line(plt.gca(), 0, 0.08, 1, 1, scale="log", label="slope 1")
    add_line(plt.gca(), 0.1, 0, 1, 0.95, "slope 0.95", 0.6, 0.5, scale="log")
    plt.subplot(122)
    for eps in [0, 0.02, 0.04]:
        plt.plot(
            t_specify,
            dict_lc["v"][0.18][eps] / t_specify,
            "o",
            label=r"$\epsilon=%g$" % eps)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$t$", fontsize="xx-large")
    plt.ylabel(r"$\xi_v/t$", fontsize="xx-large")
    add_line(plt.gca(), 0, 0.8, 1, -0.05, "slope -0.05", scale="log")

    plt.suptitle(
        r"$L=%d,\ \rho_0=%g,\ \eta=%g$" % (L, 1, 0.18), fontsize="xx-large")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig("Lc_%d.pdf" % (L))
    else:
        plt.show()
    plt.close()


def sample_ave_phi():
    """ Get smaple-averaged order parameters. """
    # epsilon = 0
    eta = np.array([0.1, 0.18, 0.35, 0.4])
    phi = np.zeros((eta.size, t_specify.size))
    for i in range(eta.size):
        t, rho_m, phi[i], r, crho_r, cv_r = sample_ave(eta[i], 0, 8192, 2)
    np.savez("phi_eps0.npz", eta=eta, t=t, phi=phi)

    # eta = 0.18
    eps = np.array([0, 0.02, 0.04, 0.06])
    phi = np.zeros((eps.size, t_specify.size))
    for i in range(eps.size):
        t, rho_m, phi[i], r, crho_r, cv_r = sample_ave(0.18, eps[i], 8192, 2)
    np.savez("phi_eta18.npz", eps=eps, t=t, phi=phi)


def plot_phi_vs_t():
    """ Plot sample-averaged order parameters against time."""
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    data = np.load("phi_eps0.npz")
    for i, eta in enumerate(data["eta"]):
        plt.plot(data["t"], data["phi"][i], "o", label=r"$\eta=%.2f$" % eta)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$t$", fontsize="x-large")
    plt.ylabel(r"$\phi$", fontsize="x-large")
    plt.legend(title=r"$\epsilon=0$")
    add_line(plt.gca(), 0, 0, 1, 1, scale="log")
    add_line(plt.gca(), 0.3, 0, 1, 1, scale="log")

    plt.subplot(122)
    data = np.load("phi_eta18.npz")
    for i, eps in enumerate(data["eps"]):
        plt.plot(
            data["t"], data["phi"][i], "o", label=r"$\epsilon=%.2f$" % eps)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$t$", fontsize="x-large")
    plt.ylabel(r"$\phi$", fontsize="x-large")
    plt.legend(title=r"$\eta=0.18$")
    add_line(plt.gca(), 0, 0, 1, 1, scale="log")

    fig.tight_layout()
    plt.show()
    plt.close()


def plot_phi_vs_xi():
    """ Plot sample-averaged order parameters against correlation lengths."""
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    data = np.load("phi_eps0.npz")
    plt.plot(dict_lc["v"][0.1], data["phi"][0], "o", label=r"$\eta=0.1$")
    plt.plot(dict_lc["v"][0.18][0], data["phi"][1], "o", label=r"$\eta=0.18$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\xi_v$", fontsize="x-large")
    plt.ylabel(r"$\phi$", fontsize="x-large")
    plt.legend(title=r"$\epsilon=0$", fontsize="large")
    add_line(plt.gca(), 0.1, 0, 1, 1, scale="log", label="slope 1")

    plt.subplot(122)
    data = np.load("phi_eta18.npz")
    plt.plot(dict_lc["v"][0.18][0], data["phi"][0], "o", label=r"$\epsilon=0$")
    plt.plot(
        dict_lc["v"][0.18][0.02],
        data["phi"][1],
        "o",
        label=r"$\epsilon=0.02$")
    plt.plot(
        dict_lc["v"][0.18][0.04],
        data["phi"][2],
        "o",
        label=r"$\epsilon=0.04$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\xi_v$", fontsize="x-large")
    plt.ylabel(r"$\phi$", fontsize="x-large")
    add_line(plt.gca(), 0.03, 0, 1, 1, scale="log")
    add_line(plt.gca(), 0.1, 0, 1, 1, scale="log")
    plt.legend(title=r"$\eta=0.18$", fontsize="large")
    fig.tight_layout()
    plt.show()
    plt.close()


def get_crossing_point(xdata, ydata, y0, scale="lin"):
    if scale == "log":
        xdata = np.log10(xdata)
        ydata = np.log10(ydata)
        y0 = np.log10(y0)
    j = -1
    for i in range(xdata.size):
        if ydata[i] > y0 and ydata[i + 1] <= y0:
            j = i
            break
    if j < 0:
        x0 = 0
    else:
        try:
            x0 = xdata[j] - (ydata[j] - y0) * (xdata[j] - xdata[j + 1]) / (
                ydata[j] - ydata[j + 1])
        except:
            return None
    if scale == "log":
        x0 = 10**x0
    return x0


if __name__ == "__main__":
    os.chdir(r"data\corr_r")
    dict_lc = {"rho": {}, "v": {}}
    dict_lc["v"][0.18] = {}
    dict_lc["v"][0.18][0] = np.array(
        [25.27, 46.45, 89.16, 171.5, 332, 657.7, 1222.5, 2412])
    # dict_lc["v"][0.18][0.02] = [
    #     25.015, 46.21, 88.02, 169.43, 327.51, 634.04, 1218.84, 2092.95
    # ]
    # dict_lc["v"][0.18][0.02] = [
    #     25.77, 47.54, 90.36, 174.04, 337.61, 657.20, 1254.14, 2201.04
    # ]
    dict_lc["v"][0.18][0.02] = [
        26.54, 48.79, 92.29, 177.57, 345.72, 674.28, 1275.55, 2249.28
    ]
    dict_lc["v"][0.18][0.04] = [
        26.50, 48.21, 91.44, 174.13, 338.28, 660.80, 1348.88, 2500.09
    ]
    dict_lc["v"][0.1] = [
        25.62, 47.87, 91.55, 177.1, 341.43, 678.6, 1238.9, 2594
    ]
    dict_lc["v"][0.35] = [25, 44.3, 84.5, 158.7, 308.1, 598, 1123, 2329]

    t_specify = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])

    varied_eta(0.18, eps=0.06, save=False, normed=False)
    # collapse6(0.18, eps=0.04, beg_idx=4, save=False, normed=False)
    # plot_lc2(save=False)
    # varied_L(2048, 4096, 8192, save=True)
    # plot_phi_vs_t()
    # plot_phi_vs_xi()
    # sample_ave_phi()
