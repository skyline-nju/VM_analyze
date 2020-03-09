"""
Output order parameter as functions of system size L and number of bands.
"""

import os
import numpy as np
import glob
# import matplotlib.pyplot as plt


def time_ave(fin, data_dict=None):
    buff = np.load(fin)
    s = os.path.basename(fin).split(".")
    L = int(s[2])
    seed = int(s[4])
    seg_phi = buff["seg_phi"]
    seg_n = buff["seg_num"]
    seg_beg = buff["seg_idx0"]
    seg_end = buff["seg_idx1"]
    phi_accum = {}
    dt_accum = {}
    tot_t = 0
    phi_mean = 0
    for i, n in enumerate(seg_n):
        dt = seg_end[i] - seg_beg[i]
        if n in phi_accum:
            phi_accum[n] += dt * seg_phi[i]
            dt_accum[n] += dt
        else:
            phi_accum[n] = dt * seg_phi[i]
            dt_accum[n] = dt
        tot_t += dt
        phi_mean += dt * seg_phi[i]
    phi_mean /= tot_t
    phi_n = {}
    rate_n = {}
    for n in phi_accum:
        phi_n[n] = phi_accum[n] / dt_accum[n]
        rate_n[n] = dt_accum[n] / tot_t
    data = {"phi_mean": phi_mean, "phi_n": phi_n, "rate_n": rate_n}
    if data_dict is not None:
        if L not in data_dict:
            data_dict[L] = {seed: data}
        else:
            data_dict[L][seed] = data


def sample_ave(phi_L_dict, rate_min=0.25):
    ave_phi_n = {}
    sample_size_n = {}
    phi_mean = 0
    seed_count = 0
    for seed in phi_L_dict:
        for n in phi_L_dict[seed]["phi_n"]:
            phi_mean += phi_L_dict[seed]["phi_mean"]
            seed_count += 1
            if phi_L_dict[seed]["rate_n"][n] > rate_min:
                phi = phi_L_dict[seed]["phi_n"][n]
                if n in ave_phi_n:
                    ave_phi_n[n] += phi
                    sample_size_n[n] += 1
                else:
                    ave_phi_n[n] = phi
                    sample_size_n[n] = 1
    for n in ave_phi_n.keys():
        ave_phi_n[n] /= sample_size_n[n]
    phi_mean /= seed_count
    return ave_phi_n, phi_mean


if __name__ == "__main__":
    os.chdir(r"G:\data\band\Lx\snapshot")
    path1 = "eps20_mpi/"
    path2 = "eps20_2019/"
    path3 = "eps20_tanglou/"
    path4 = "eps20/"

    pat = "mb_350.20.*.200.*.npz"
    files = glob.glob(path1 + pat) + glob.glob(path2 + pat) +\
        glob.glob(path3 + pat) + glob.glob(path4 + pat)

    phi_LSN = {}
    for f in files:
        time_ave(f, phi_LSN)

    L_arr = np.array([i for i in sorted(phi_LSN.keys())])
    print(L_arr)

    phi_mean = np.zeros(L_arr.size)
    phi_LN = {}
    for i, L in enumerate(L_arr):
        phi_LN[L], phi_mean[i] = sample_ave(phi_LSN[L])
    with open("phi_L_eta350_eps20.dat", "w") as f:
        for L in L_arr:
            for n in sorted(phi_LN[L].keys()):
                if n >= 1:
                    f.write("%d\t%d\t%.8f\n" % (L, n, phi_LN[L][n]))

    with open("raw_0.35_0.02.dat", "w") as f:
        for i, L in enumerate(L_arr):
            f.write("%d\t%.8f\n" % (L, phi_mean[i]))
