import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def plot_3x3(L, seed=20150, eta=0.18, disorder_t="RT"):
    os.chdir(r"E:\data\random_torque\replica2\L=%d" % L)
    if L == 256 or L == 512:
        pat = "p%d.%g.%g.%d.%03d.dat"
        theta0_arr = [0, 60, 120, 180, 240, 300]
    else:
        pat = "p%d.%g.%g.%d_%03d.dat"
        theta0_arr = [0, 90, 180, 270]

    fig = plt.figure(figsize=(8, 12), constrained_layout=True)
    eps_arr = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    for i, eps in enumerate(eps_arr):
        ax = fig.add_subplot(3, 3, i + 1, projection="polar")
        for theta0 in theta0_arr:
            file_in = pat % (L, eta * 1000, eps * 1000, seed, theta0)
            print(file_in)
            phi, theta = read_phi_theta(file_in, 0)
            x = (np.arange(phi.size) + 1) * 100
            theta = untangle(theta)
            ax.plot(theta, x)
            ax.set_title(
                r"(%s) $\epsilon=%.3f$" % (alphabet[i], eps),
                fontsize="large",
                color="r",
                pad=11)
    plt.suptitle(
        r"RT: $L=%d, \eta=%g,$ seed=%d" % (L, eta, seed), fontsize="x-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    disorder_t = "RT"
    L = 2048
    eta = 0.18
    eps = 0.055
    seed = 20200712

    os.chdir(r"E:\data\random_torque\replica2\L=%d" % L)
    if L == 256:
        pat = "p%d.%g.%g.%d.%03d.dat"
    else:
        pat = "p%d.%g.%g.%d.%03d.dat"

    # theta0_arr = [0, 60, 120, 180, 240, 300]
    theta0_arr = [0, 180]
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = plt.subplot(111, projection="polar")
    for theta0 in theta0_arr:
        file_in = pat % (L, eta * 1000, eps * 1000, seed, theta0)
        phi, theta = read_phi_theta(file_in, 0)
        x = (np.arange(phi.size) + 1) * 100
        theta = untangle(theta)
        ax.plot(theta, x)
    plt.show()
    plt.close()
    # plot_3x3(512, seed=20150)
