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


def read(file, load_Sk=False):
    """ Read a file contaning Cr (and Sk).

        File name looks like cr_%g_%g_%g_%d_%d_*.bin or
        crsk_%g_%g_%g_%d_%d_*.bin.
    """
    s = file.replace(".bin", "").split("_")
    size_r = int(s[-2])
    tot_frames = int(s[-1])
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
        for i in range(tot_frames):
            buff = f.read(4)
            t[i], = struct.unpack("i", buff)
            buff = f.read(24)
            rho_m[i], vx_m[i], vy_m[i] = struct.unpack("ddd", buff)
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


def time_ave(t, rho_m, vx_m, vy_m, crho_r, cv_r, srho_k=None, sv_k=None):
    """ Average over nearby frames. """
    pos = split_array(t)
    n = len(pos) - 1
    t_new = np.zeros(n)
    rho_m_new = np.zeros(n)
    vx_m_new = np.zeros(n)
    vy_m_new = np.zeros(n)
    crho_r_new = np.zeros((n, crho_r.shape[1]))
    cv_r_new = np.zeros_like(crho_r_new)
    if srho_k is not None:
        srho_k_new = np.zeros_like(crho_r_new)
        sv_k_new = np.zeros_like(crho_r_new)
    for i in range(n):
        j, k = pos[i], pos[i + 1]
        t_new[i] = np.mean(t[j:k])
        rho_m_new[i] = np.mean(rho_m[j:k])
        vx_m_new[i] = np.mean(vx_m[j:k])
        vy_m_new[i] = np.mean(vy_m[j:k])
        crho_r_new[i] = np.mean(crho_r[j:k], axis=0)
        cv_r_new[i] = np.mean(cv_r[j:k], axis=0)
        if srho_k is not None:
            srho_k_new[i] = np.mean(srho_k[j:k], axis=0)
            sv_k_new[i] = np.mean(sv_k[j:k], axis=0)
    if srho_k is None:
        return t_new, rho_m_new, vx_m_new, vy_m_new, crho_r_new, cv_r_new
    else:
        return t_new, rho_m_new, vx_m_new, vy_m_new, crho_r_new, cv_r_new, \
            srho_k_new, sv_k_new


def sample_ave(eta, eps, L, l, rho0=1, t_ave=True, files=None):
    """ Average over samples. """
    if files is None:
        files = glob.glob("cr*_%g_%g_%g_%d_%d_*.bin" % (eta, eps, rho0, L, l))
    t, rho_m, vx_m, vy_m, r, crho_r, cv_r = read(files[0])
    for file in files[1:]:
        print(file)
        t, rho_m2, vx_m2, vy_m2, r, crho_r2, cv_r2 = read(file)
        rho_m += rho_m2
        vx_m += vx_m2
        vy_m += vy_m2
        crho_r += crho_r2
        cv_r += cv_r2
    if len(files) > 1:
        rho_m /= len(files)
        vx_m /= len(files)
        vy_m /= len(files)
        crho_r /= len(files)
        cv_r /= len(files)
    if t_ave:
        t, rho_m, vx_m, vy_m, crho_r, cv_r = time_ave(t, rho_m, vx_m, vy_m,
                                                      crho_r, cv_r)
    return t, rho_m, vx_m, vy_m, r, crho_r, cv_r


def plot_cr(t,
            r,
            cr,
            start_idx=0,
            lc=None,
            mean=0,
            xlog=True,
            ylog=True,
            t_specify=None,
            ax=None):
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    else:
        flag_show = False

    for i, cr_t in enumerate(cr[start_idx:]):
        i += start_idx
        if t_specify is None or t[i] in t_specify:
            if lc is None:
                r_new = r
            else:
                r_new = r / lc[i]
            ax.plot(r_new, cr_t - mean, "o", label="%g" % (t[i]), ms=1)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if flag_show:
        ax.legend(title=r"$t=$")
        plt.show()
        plt.close()


if __name__ == "__main__":
    os.chdir(r"data\corr_r")
    t_specify = [25, 50, 100, 200, 400, 800, 1600, 3200]
    # lc = [8.39, 17.4, 37.14, 73.46, 160, 314, 609, 1106, 2509]
    # lc = [35, 55, 95, 172, 331, 640, 1133, 2268]
    # lc = [24.3, 44, 83, 158, 305, 597, 1110, 2424]
    lc_rho = [8.34, 18.27, 36.68, 76.24, 157.8, 312.6, 627.6, 1149]
    lc_v = [6.24, 10.12, 17.24, 30.87, 56, 105, 202.8, 387]
    eta = 0.18
    eps = 0
    rho0 = 1
    # L = 4096
    L = 8192
    l = 2
    # files = glob.glob("cr_%g_%g_%g_%d_%d_*.bin" % (eta, eps, rho0, L, l))
    t, rho_m, vx_m, vy_m, r, crho_r, cv_r = sample_ave(eta, eps, L, l)
    # ax = plt.subplot(111)
    plot_cr(t, r, cv_r, lc=lc_v, xlog=False, start_idx=3)
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    # ax = plt.subplot(111)
    # ax.loglog(t[1:], lc_rho[1:], 'o', t[1:], lc_v[1:], 's')
    # add_line(ax, 0, 0.12, 1, slope=1, scale="log")
    # add_line(ax, 0, 0, 1, slope=0.9, scale="log")
    # add_line(ax, 0.1, 0, 1, slope=1, scale="log")
    # plt.show()
    # plt.close()
