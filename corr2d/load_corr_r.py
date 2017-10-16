""" Load the binary file where the spherically averaged correlation functions
    for density and velocity are saved.
"""

import struct
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def read(file):
    s = file.replace(".bin", "").split("_")
    size_r = int(s[-2])
    tot_frames = int(s[-1])
    t = np.zeros(tot_frames, int)
    rho_m = np.zeros(tot_frames)
    vx_m = np.zeros_like(rho_m)
    vy_m = np.zeros_like(rho_m)
    crho_r = np.zeros((tot_frames, size_r))
    cv_r = np.zeros_like(crho_r)

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
    return t, rho_m, vx_m, vy_m, r, crho_r, cv_r


def sample_average(eps,
                   L,
                   l,
                   eta=0.18,
                   rho0=1,
                   time_averaged=False,
                   ax=None,
                   xi=None):
    os.chdir(r"D:\code\VM\VM\corr_r")
    files = glob.glob("cr_%g_%g_%g_%d_%d_*.bin" % (eta, eps, rho0, L, l))
    s = files[0].replace(".bin", "").split("_")
    tot_frames = int(s[-1])
    len_r = int(s[-2])
    crho_r_m = np.zeros((tot_frames, len_r))
    cv_r_m = np.zeros((tot_frames, len_r))
    for file in files:
        t, rho_m, vx_m, vy_m, r, crho_r, cv_r = read(file)
        crho_r_m += crho_r
        cv_r_m += cv_r
    crho_r_m /= len(files)
    cv_r_m /= len(files)

    if time_averaged:
        crho_r_new = []
        cv_r_new = []
        t_new = []
        t_pre = t[0]
        i_beg = 0
        for i, t_cur in enumerate(t):
            if t_cur - t_pre > 1:
                t_new.append(np.mean(t[i_beg:i]))
                crho_r_new.append(np.mean(crho_r_m[i_beg:i], axis=0))
                cv_r_new.append(np.mean(cv_r_m[i_beg:i], axis=0))
                i_beg = i
            elif i == t.size - 1:
                t_new.append(np.mean(t[i_beg:i + 1]))
                crho_r_new.append(np.mean(crho_r_m[i_beg:i + 1], axis=0))
                cv_r_new.append(np.mean(cv_r_m[i_beg:i + 1], axis=0))
            t_pre = t_cur
        crho_r_m = np.array(crho_r_new)
        cv_r_m = np.array(cv_r_new)
        t = np.array(t_new)
        print(t_new)
    print(crho_r_m.shape)

    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False
    for i, cr in enumerate(cv_r_m):
        if xi is not None:
            rnew = r / xi[i]
        else:
            rnew = r
        y = cr - 1
        ax.plot(rnew, y, "o", ms=2, label="%g" % t[i])
    # ax.set_yscale("log")
    ax.set_xscale("log")
    # ax.set_ylim(1e-4)
    ax.legend(loc="upper right", title=r"$t=$")
    if flag_show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    # xi1 = [8.39, 17.4, 37.14, 73.46, 160, 314, 609, 1106, 2509]
    xi1 = [35, 55, 95, 172, 331, 640, 1133, 2268]
    xi2 = [24.3, 44, 83, 158, 305, 597, 1110, 2424]
    sample_average(0, 8192, 2, eta=0.18, time_averaged=True, ax=ax1, xi=xi1)
    sample_average(0, 8192, 2, eta=0.35, time_averaged=True, ax=ax2, xi=xi2)
    plt.show()
    plt.close()
    plt.close()

    t = [25, 50, 100, 200, 400, 800, 1600, 3200]
    plt.loglog(t, xi2, "o")
    plt.show()
