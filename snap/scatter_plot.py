import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from load_snap import RawSnap


def plot_scatter(filename):
    snap = RawSnap(filename)
    x, y, theta = snap.one_frame()
    s = os.path.basename(filename).replace(".bin", "").split("_")
    Lx = int(s[1])
    eta = float(s[2])
    eps = float(s[3])
    seed = int(s[4])
    t = int(s[5])

    theta = theta / np.pi * 180
    theta[theta < 0] += 360
    plt.scatter(x, y, s=1, c=theta, marker=1, cmap="hsv")
    cb = plt.colorbar()
    cb.set_label("orientation", fontsize="large")
    plt.axis("scaled")
    plt.xlim(0, Lx)
    plt.ylim(0, Lx)
    plt.xlabel(r"$x$", fontsize="x-large")
    plt.ylabel(r"$y$", fontsize="x-large")
    plt.title(r"$L=%d,\ \eta=%g,\ \epsilon=%g,\ {\rm seed}=%d,\ t=%d$" % (
            Lx, eta, eps, seed, t))


if __name__ == "__main__":
    eta = 0.05
    os.chdir(r"D:\data\random_torque\snapshot\eta=%g" % eta)
    files = glob.glob("*.bin")
    for file in files:
        if not os.path.exists(file.replace(".bin", ".png")):
            ax = plt.subplot(111)
            plot_scatter(file, ax)
            plt.show()
            plt.close()
