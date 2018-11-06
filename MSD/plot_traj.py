import numpy as np
import matplotlib.pyplot as plt
import os


def read_traj(eta, eps, L, seed=123, disorder="RT"):
    filename = "traj_%s_%d_%g_%g_%d.dat" % (disorder.lower(), L, eta, eps, seed)
    with open(filename) as f:
        lines = f.readlines()
        nrow = len(lines)
        t = np.zeros(nrow, int)
        x = np.zeros((4, nrow))
        y = np.zeros((4, nrow))
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            t[i] = int(s[0])
            x[0, i] = float(s[1])
            y[0, i] = float(s[2])
            x[1, i] = float(s[3])
            y[1, i] = float(s[4])
            x[2, i] = float(s[5])
            y[2, i] = float(s[6])
            x[3, i] = float(s[7])
            y[3, i] = float(s[8])
    return t, x, y


if __name__ == "__main__":
    os.chdir(r"D:\data\random_field\normalize_new\MSD_new")
    t, x, y = read_traj(0, 0.09, 32)
    plt.scatter(x[3], y[3], s=0.01, c=t)
    cb = plt.colorbar()
    cb.set_label(r"$t$", fontsize="x-large")
    plt.show()
    plt.close()
