import glob
import numpy as np
# import matplotlib.pyplot as plt
import sys
import random
import os

sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def read_replicas(L,
                  eta,
                  eps,
                  seed,
                  data_dir,
                  angles=[0, 90, 180, 270],
                  ret_var=False):
    phi_mean = np.zeros(len(angles))
    if ret_var:
        phi_var = np.zeros_like(phi_mean)
    for i, angle in enumerate(angles):
        fname = "%s\\p%d.%g.%g.%d.%03d.dat" % (data_dir, L, eta * 1000,
                                               eps * 1000, seed, angle)
        phi, theta = read_phi_theta(fname, ncut=10000)
        phi_mean[i] = np.mean(phi)
        if ret_var:
            phi_var[i] = np.var(phi)
    if not ret_var:
        return phi_mean
    else:
        return phi_mean, phi_var


def get_phi_mean(L, eta, eps, angles=[0, 90, 180, 270], save=False):
    data_dir = "D:\\data\\VM2d\\random_torque\\replica\\serials"
    pat = "%s\\p%d.%g.%g.*.dat" % (data_dir, L, eta * 1000, eps * 1000)
    files = glob.glob(pat)
    seed_list = []
    for f in files:
        seed = int(os.path.basename(f).split(".")[3])
        if seed not in seed_list:
            seed_list.append(seed)
    n_samples = len(seed_list)
    print("sample size =", n_samples)
    phi_mean = np.zeros((n_samples, len(angles)))
    phi_var = np.zeros_like(phi_mean)
    for i, seed in enumerate(seed_list):
        phi_mean[i], phi_var[i] = read_replicas(
            L, eta, eps, seed, data_dir, angles, ret_var=True)
    if save:
        out_dir = "D:\\data\\VM2d\\random_torque\\replica"
        fout = "%s\\%d_%g_%g.npz" % (out_dir, L, eta, eps)
        np.savez(fout, phi_mean=phi_mean, phi_var=phi_var)
    else:
        return phi_mean


def cal_var(L, eta, eps, seed):
    random.seed(seed)
    data_dir = "D:\\data\\VM2d\\random_torque\\replica"
    fin = "%s\\%d_%g_%g.npz" % (data_dir, L, eta, eps)
    npzfile = np.load(fin)
    phi_mean = npzfile["phi_mean"]
    phi_var = npzfile["phi_var"]
    n_samples, n_replica = phi_mean.shape
    phi_max, phi_rand = np.zeros((2, n_samples))
    var_max, var_rand = np.zeros((2, n_samples))
    for i in range(n_samples):
        phi_max[i] = np.max(phi_mean[i])
        phi_rand[i] = random.choice(phi_mean[i])
        j1 = np.argwhere(phi_mean[i] == phi_max[i])[0][0]
        j2 = np.argwhere(phi_mean[i] == phi_rand[i])[0][0]
        var_max[i] = phi_var[i][j1]
        var_rand[i] = phi_var[i][j2]
    print(
        np.mean(phi_max), np.mean(phi_rand), np.mean(var_max),
        np.mean(var_rand), np.var(phi_max), np.var(phi_rand))


if __name__ == "__main__":
    L = 512
    eta = 0.18
    eps = 0.035
    # get_phi_mean(L, eta, eps, save=True)
    # print(phi_mean.shape)
    for L in [180, 256, 362, 512]:
        if L == 256:
            seed = 9
        elif L == 362:
            seed = 5
        else:
            seed = 1
        cal_var(L, eta, eps, seed)
