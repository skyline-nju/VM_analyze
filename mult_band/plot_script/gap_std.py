""" Plot standard deviation of gaps between two nearest bands."""

import numpy as np
import os
import matplotlib.pyplot as plt
from read_npz import eq_Lx_and_nb, read_matched_file


def plot_phi_vs_std_gap(Lx, nb, marker, eta=350, eps=20, ax=None):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    dictLSN = read_matched_file({"Lx": Lx})
    f = eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)
    phi = np.array([i for i in f])
    f = eq_Lx_and_nb(Lx, nb, "std_gap", dictLSN=dictLSN)
    std_gap = np.array([i for i in f])
    # f = eq_Lx_and_nb(Lx, nb, "rate", dictLSN=dictLSN)
    # rate = np.array([i for i in f])
    ax.plot(std_gap, phi, marker)
    # ax.scatter(std_gap, phi, c=rate)

    if flag_show:
        plt.show()
        plt.close()
    else:
        return np.mean(phi), np.mean(std_gap)


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    Lxs = [460]
    mk = ["o", "s", ">", "h", "p", "<", "^", "v"]
    ax = plt.subplot(111)
    phi, std_gap = np.zeros((2, len(Lxs)))
    for i, Lx in enumerate(Lxs):
        phi[i], std_gap[i] = plot_phi_vs_std_gap(Lx, 2, mk[i], ax=ax)
    # plt.legend(loc="best")
    plt.show()
    plt.close()

    Lx = np.array(Lxs)
    plt.subplot(121)
    plt.plot(std_gap, phi, "-o")
    plt.subplot(122)
    plt.plot(Lx, std_gap, "-o")
    plt.show()
    plt.close()
