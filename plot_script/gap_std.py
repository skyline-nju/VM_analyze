""" Plot standard deviation of gaps between two nearest bands."""

import numpy as np
import os
import matplotlib.pyplot as plt
from read_npz import fixed_para


def plot_phi_vs_std_gap(Lx, nb, marker, eta=350, eps=20, ax=None):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    f = fixed_para("mean_phi", "std_gap", Lx=Lx, nb=nb)
    for phi, std_gap in f:
        plt.plot(std_gap/Lx, phi, marker, label=r"$L_x=%d, n_b=%d$" % (Lx, nb))

    if flag_show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    Lxs = [400, 420, 440, 460, 480]
    mk = ["ko", "rs", "g>", "b<", "yp"]
    ax = plt.subplot(111)
    for i, Lx in enumerate(Lxs):
        plot_phi_vs_std_gap(Lx, 2, mk[i], ax=ax)
    # plt.legend(loc="best")
    plt.show()
    plt.close()

