import numpy as np
import os


def cal_phi(infile, n_cut):
    with open(infile) as f:
        lines = f.readlines()[n_cut:]
        phi = np.array([float(line.split("\t")[0]) for line in lines])
        return np.mean(phi)


def varied_domain_size():
    L = [16, 22, 32, 46, 64, 80, 96, 120]
    n_cut = 2500
    eps = 0.02
    seed = 41
    for l in L:
        filename = "phi_%d_0.20_%.3f_1.0_%d.dat" % (l, eps, seed)
        if os.path.exists(filename):
            print(l, cal_phi(filename, n_cut))


def varied_eps():
    L = 80
    n_cut = 2500
    eps = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12,
           0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
    seed = 11
    for epsilon in eps:
        filename = "phi_%d_0.20_%.3f_1.0_%d.dat" % (L, epsilon, seed)
        if os.path.exists(filename):
            print(epsilon, cal_phi(filename, n_cut))


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d")
    varied_eps()
    # varied_domain_size()
