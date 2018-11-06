import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def read_phi(filename, ncut):
    with open(filename) as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(i.split("\t")[0]) for i in lines])
        phi_mean = np.mean(phi)
    return phi_mean


def read_all(L, eta, eps, rho0=1.):
    files = glob.glob("phi_%d_%.2f_%.3f_%.1f_*.dat" % (L, eta, eps, rho0))
    phi_arr = np.zeros(len(files))
    for i, file_i in enumerate(files):
        phi_arr[i] = read_phi(file_i, 2000)
    with open(r"..\phi3_%d_%.2f_%.3f.dat" % (L, eta, eps), "w") as f:
        for phi in phi_arr:
            f.write("%.8f\n" % phi)


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d\order_para")
    read_all(22, 0.2, 0.12)
