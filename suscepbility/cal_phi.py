#!/home-gk/users/nscc1185/Applications/anaconda3/bin/python
# import glob
import numpy as np


def read(file, ncut=2000):
    with open(file) as f:
        lines = f.readlines()[ncut:]
        phi = np.array([float(line.split("\t")[0]) for line in lines])
        mean = np.mean(phi)
        var = np.var(phi)
    return mean, var


def read_files(L, seed, eps):
    phi = np.zeros(len(eps))
    xi = np.zeros_like(phi)
    for i, epsilon in enumerate(eps):
        file = "p%d.180.%d.%d.dat" % (L, epsilon, seed)
        phi[i], xi[i] = read(file)
    return phi, xi * L * L


def write_data(L, n):
    eps = [500, 550, 575, 600, 625, 650, 675, 700, 725, 750, 800, 850]
    phi = np.zeros((n, len(eps)))
    xi = np.zeros((n, len(eps)))
    for i in range(n):
        seed = 1080000 + i
        print("seed = %d" % seed)
        phi[i], xi[i] = read_files(L, seed, eps)
    np.savez("%d.npz" % L, phi=phi, xi=xi, eps=eps)


if __name__ == "__main__":
    write_data(90, 200)
