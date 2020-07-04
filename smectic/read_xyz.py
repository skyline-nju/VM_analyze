import numpy as np
import os
import matplotlib.pyplot as plt


def read_xyz(fin):
    with open(fin, "r") as f:
        f.seek(0, 2)
        filesize = f.tell()
        print("filesize", filesize)
        f.seek(0)
        while f.tell() < filesize:
            n = int(f.readline().rstrip("\n"))
            f.readline()
            x, y, theta = np.zeros((3, n))
            for i in range(n):
                s = f.readline().rstrip("\n").split("\t")
                x[i], y[i], theta[i] = float(s[1]), float(s[2]), float(s[3])
            yield x, y, theta


if __name__ == "__main__":
    folder = r"D:\data\smectic\New Folder\rho1"
    # folder = r"D:\data\smectic"

    os.chdir(folder)
    fname = "s0.02_b0.5_r1.5_L40.extxyz"
    # fname = "s0.02_b1_r10_L40.extxyz"
    frames = read_xyz(fname)
    for (x, y, theta) in frames:
        vx = np.cos(theta)
        vy = np.sin(theta)
        vxm = np.mean(vx)
        vym = np.mean(vy)
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.plot(x, y, ".", ms=5)
        xc, yc = x[2], y[2]
        phi = np.linspace(0, np.pi * 2, 50)
        ax.plot(xc + np.cos(phi), yc + np.sin(phi), "r--")
        # ax.arrow(x, y, vx * 0.1, vy * 0.1)
        for i in range(x.size):
            ax.arrow(x[i], y[i], vx[i] * 0.5, vy[i] * 0.5)
        ax.arrow(20, 20, vxm * 10, vym * 10, color="r")
        ax.arrow(20, 20, vym * 10, -vxm * 10, color="g")
        # theta_c = np.arctan2(np.sum(vy), np.sum(vx))
        # hist, bin_edges = np.histogram(
        #     theta, bins=360, range=(-np.pi, np.pi), density=True)
        # bin_c = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # plt.plot(bin_c, hist, "-")
        # plt.axvline(x=theta_c, color="r")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        plt.show()
        plt.close()

