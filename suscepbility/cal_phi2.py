#!/home-gk/users/nscc1185/Applications/anaconda3/bin/python
import numpy as np
import os
import glob
import sys


def read(file, ncut=2000):
    with open(file) as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(line.split("\t")[0]) for line in lines])
        mean = np.mean(phi)
        var = np.var(phi)
    return mean, var


def cal_phi(L, eps, beg=None, end=None):
    os.chdir("%d" % L)
    if beg is None:
        files = glob.glob("p%d.180.%d.*.dat" % (L, eps))
    else:
        files = [
            "p%d.180.%d.%d.dat" % (L, eps, (L + 18) * 10000 + j)
            for j in range(beg, end)
        ]
    phi = []
    chi = []
    print("%d files" % len(files))
    for file in files:
        print(file)
        try:
            mean, var = read(file)
            phi.append(mean)
            chi.append(var)
        except:
            print(file)
    phi = np.array(phi)
    chi = np.array(chi) * L ** 2
    print("%d\t%f\t%f\t%d\t%f\t%f\n" % (L, np.mean(phi), np.std(phi), phi.size,
                                        np.mean(chi), np.std(chi)))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        cal_phi(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 5:
        cal_phi(
            int(sys.argv[1]),
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
