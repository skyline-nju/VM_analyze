""" Read the spherially averaged correlation functions and structure factors.
"""

import struct
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from add_line import add_line


def read(file):
    print("file: ", file)
    s = file.replace(".bin", "").split("_")
    size_r = int(s[-2])
    tot_frames = int(s[-1])
    t = np.zeros(tot_frames, int)
    rho_m = np.zeros(tot_frames)
    vx_m = np.zeros_like(rho_m)
    vy_m = np.zeros_like(rho_m)
    crho_r = np.zeros((tot_frames, size_r))
    cv_r = np.zeros_like(crho_r)
    sk_rho = np.zeros_like(crho_r)
    sk_v = np.zeros_like(crho_r)

    f = open(file, "rb")
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
        buff = f.read(8 * size_r)
        sk_rho[i] = np.array(struct.unpack("%dd" % size_r, buff))
        buff = f.read(8 * size_r)
        sk_v[i] = np.array(struct.unpack("%dd" % size_r, buff))
    f.close()
    return t, rho_m, vx_m, vy_m, r, crho_r, cv_r, sk_rho, sk_v


def sample_average(eps, L, l, eta=0.18, rho0=1, t_ave=False):
    os.chdir(r"D:\code\VM\VM\corr_r")
    files = glob.glob("crsk_%g_%g_%g_%d_%d_*.bin" % (eta, eps, rho0, L, l))
    s = files[0].replace(".bin", "").split("_")
    tot_frames = int(s[-1])
    len_r = int(s[-2])
    crho_r_m = np.zeros((tot_frames, len_r))
    cv_r_m = np.zeros((tot_frames, len_r))
    sk_rho_m = np.zeros((tot_frames, len_r))
    sk_v_m = np.zeros((tot_frames, len_r))
    for file in files:
        t, rho_m, vx_m, vy_m, r, crho_r, cv_r, sk_rho, sk_v = read(file)
        crho_r_m += crho_r
        cv_r_m += cv_r
        sk_rho_m += sk_rho
        sk_v_m += sk_v
    crho_r_m /= len(files)
    cv_r_m /= len(files)

    if t_ave:
        crho_r_new = []
        cv_r_new = []
        sk_rho_new = []
        sk_v_new = []
        t_new = []
        t_pre = t[0]
        i_beg = 0
        for i, t_cur in enumerate(t):
            if t_cur - t_pre > 1:
                t_new.append(np.mean(t[i_beg:i]))
                crho_r_new.append(np.mean(crho_r_m[i_beg:i], axis=0))
                cv_r_new.append(np.mean(cv_r_m[i_beg:i], axis=0))
                sk_rho_new.append(np.mean(sk_rho_m[i_beg:i], axis=0))
                sk_v_new.append(np.mean(sk_v_m[i_beg:i], axis=0))
                i_beg = i
            elif i == t.size - 1:
                t_new.append(np.mean(t[i_beg:i + 1]))
                crho_r_new.append(np.mean(crho_r_m[i_beg:i + 1], axis=0))
                cv_r_new.append(np.mean(cv_r_m[i_beg:i + 1], axis=0))
                sk_rho_new.append(np.mean(sk_rho_m[i_beg:i + 1], axis=0))
                sk_v_new.append(np.mean(sk_v_m[i_beg:i + 1], axis=0))
            t_pre = t_cur
        crho_r_m = np.array(crho_r_new)
        cv_r_m = np.array(cv_r_new)
        sk_rho_m = np.array(sk_rho_new)
        sk_v_m = np.array(sk_v_new)
        t = np.array(t_new)
    return t, r, crho_r_m, cv_r_m, sk_rho_m, sk_v_m


if __name__ == "__main__":
    t, r, crho_r_m, cv_rm, sk_rho_m, sk_v_m = sample_average(0, 4096, 2, eta=0.18, t_ave=True)
    for i, sk in enumerate(sk_v_m[2:]):
        plt.plot(r, sk, "o", ms=2, label="%d" % t[i+2])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10)
    plt.legend()
    add_line(plt.gca(), 0.5, 0.5, 1, -2, scale="log")
    add_line(plt.gca(), 0.5, 0.5, 1, -1, scale="log")    
    plt.show()
    plt.close()
