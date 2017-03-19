""" Plot rho_gas vs. phi. """

import os
import numpy as np
import matplotlib.pyplot as plt
from read_npz import read_matched_file, eq_Lx_and_nb


def get_data(nb, Lx=None):
    if Lx is None:
        os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps0")
        if nb == 2:
            Lxs = range(320, 480, 20)
        elif nb == 3:
            Lxs = range(480, 740, 20)
        elif nb == 4:
            Lxs = range(680, 940, 20)
        elif nb == 5:
            Lxs = range(800, 1020, 20)
        else:
            print("nb is too large")
            return
        phi = []
        peak = []
        Lx = []
        v = []

        for i, Lx0 in enumerate(Lxs):
            dictLSN = read_matched_file({"Lx": Lx0})
            seed = sorted(dictLSN[Lx0].keys())[0]
            if nb in dictLSN[Lx0][seed]:
                phi.append(dictLSN[Lx0][seed][nb]["mean_phi"])
                peak.append(dictLSN[Lx0][seed][nb]["ave_peak"])
                v.append(dictLSN[Lx0][seed][nb]["mean_v"])
                Lx.append(Lx0)
        phi = np.array(phi)
        peak = np.array(peak)
        v = np.array(v)
        Lx = np.array(Lx)
        return phi, peak, v, Lx
    else:
        if 400 <= Lx <= 480:
            os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")
        else:
            os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps20")
        dict0 = read_matched_file({"Lx": Lx})
        phi = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dict0)])
        peak = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dict0)])
        v = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "mean_v", dictLSN=dict0)])
        return phi, peak, v


def plot_eq_nb_and_Lx(nb, Lx):
    """ Plot rho_gas vs phi with fixed nb and Lx. """

    phi, peak, v = get_data(nb, Lx)
    cList = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])
    plt.subplot(121)
    x = np.arange(Lx) + 0.5
    for i, pk in enumerate(peak):
        plt.plot(x, pk, c=cList[i])

    plt.subplot(122)
    for xmax in range(200, 250, 10):
        rho_gas = np.mean(peak[:, 190:xmax], axis=1)
        rho_exc = 1 - rho_gas
        z = np.polyfit(rho_exc, phi, 1)
        plt.plot(
            rho_exc,
            phi,
            "o",
            label=r"$x_{max}=%d, \rm{slope}=%g$" % (xmax, z[0]))

    plt.legend(loc="best")
    plt.show()
    plt.close()
    print("velocity: mean=%g, std=%g" % (v.mean(), v.std()))


def case_eps0_eq_nb(nb):
    """ Plot rho_gas vs phi with fixed nb for disorder free case. """

    def get_rho_gas(peak, min=190, max=210):
        rho_gas = np.zeros(peak.size)
        for i, pk in enumerate(peak):
            rho_gas[i] = np.mean(pk[min:max])
        return rho_gas

    phi, peak, v, Lx = get_data(nb)
    plt.subplot(121)
    for i, pk in enumerate(peak):
        x = np.arange(Lx[i]) + 0.5
        plt.plot(x, pk)

    plt.subplot(122)
    xmin = 190
    for xmax in range(200, 240, 5):
        rho_gas = get_rho_gas(peak, xmin, xmax)
        rho_exc = 1 - rho_gas
        z = np.polyfit(rho_exc, phi, 1)
        print(z[0] * 0.5, np.mean(v))
        plt.plot(rho_exc, phi, "-o", label=r"$x\in[%d: %d]$" % (xmin, xmax))
    plt.legend(loc="best")
    plt.show()
    plt.close()

    plt.plot(Lx, v, "-s")
    plt.show()


def two_panel():
    def one_axes(phi, peak, ax, xmin=190, xmax=210, v=None, v0=0.5,
                 marker="o"):
        rho_gas = np.array([pk[xmin:xmax].mean() for pk in peak])
        rho_exc = 1 - rho_gas
        ax.plot(rho_exc, phi * v0, marker)
        if v is not None:
            ax.plot(rho_exc, rho_exc * v.mean())

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    phi, peak, v = get_data(2, 440)
    one_axes(phi, peak, ax1, v=v, marker="bo")
    phi, peak, v = get_data(3, 660)
    one_axes(phi, peak, ax1, marker="rs")
    phi, peak, v = get_data(4, 880)
    one_axes(phi, peak, ax1, marker="g>")

    ax2 = plt.subplot(122)
    phi, peak, v, Lx = get_data(2)
    one_axes(phi, peak, ax2, v=v)
    plt.show()


if __name__ == "__main__":
    # case_eps0_eq_nb(2)
    # plot_eq_nb_and_Lx(2, 440)
    two_panel()
