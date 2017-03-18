''' plot time-averaged density profile for epsilon=0.02 cases. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from axes_zoom_effect import zoom_effect03
from read_npz import read_matched_file, eq_Lx_and_nb


def plot_rho_gas_vs_phi(Lx, nb, eta=350, eps=20):
    def one_Lx(Lx0, nb0, maker):
        if Lx0 == 440:
            os.chdir(path2)
        else:
            os.chdir(path1)
        rho_0 = 1
        dictLSN = read_matched_file({"Lx": Lx0, "eps": eps, "eta": eta})
        f = eq_Lx_and_nb(Lx0, nb0, "mean_phi", dictLSN=dictLSN)
        phi = np.array([i for i in f])
        f = eq_Lx_and_nb(Lx0, nb0, "ave_peak", dictLSN=dictLSN)
        rho_gas = np.array([peak[190:230].mean() for peak in f])
        ax.plot(
            rho_0 - rho_gas,
            phi,
            maker,
            label=r"$L_x=%d, n_b=%d$" % (Lx0, nb0))
        z = np.polyfit(rho_0 - rho_gas, phi, 1)
        print(z)

    path1 = "E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps20"
    path2 = "E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband"

    ax = plt.subplot(111)
    mk = ["ro", "bs", "g>", "k<"]
    if isinstance(Lx, list):
        for i, Lx0 in enumerate(Lx):
            one_Lx(Lx0, nb[i], mk[i])
    else:
        one_Lx(Lx, nb)

    plt.legend(loc="best")
    plt.show()
    plt.close()


def plot_peak(Lx, nb, eta=350, eps=20):
    """ Plot time-averaged peaks for differernt samples with zoom effect."""

    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    fig = plt.figure(1, figsize=(7, 6))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)

    ax1.set_xlim(60, 240)
    ax3.set_xlim(190, 230)
    ax3.set_ylim(0.45, 0.5)
    zoom_effect03(ax3, ax1, 190, 230, loc="downward")

    dictLSN = read_matched_file({"Lx": Lx, "eta": eta, "eps": eps})
    phi = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)])
    color_list = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])

    x = np.arange(Lx) + 0.5
    rho_gas = np.zeros_like(phi)
    for i, peak in enumerate(
            eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dictLSN)):
        ax1.plot(x, peak, c=color_list[i], lw=0.8)
        ax3.plot(x, peak, c=color_list[i])
        rho_gas[i] = np.mean(peak[190:230])
    ax2.axis("auto")
    sca = ax2.scatter(rho_gas, phi, c=phi, cmap="jet")
    z = np.polyfit(rho_gas, phi, 1)
    print(z)

    ax1.set_title(r"$(a)$")
    ax2.set_title(r"$(b)$")
    ax3.set_title(r"$(c)$")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\eta=%g,\epsilon=%g, \rho_0=1, L_x=%d, L_y=200, n_b=%d$" %
                 (eta / 1000, eps / 1000, Lx, nb))
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.03, 0.78])
    cb = fig.colorbar(sca, cax=cbar_ax)
    cb.set_label(r"$\langle \phi \rangle_t$")

    plt.show()
    # plt.savefig(r"E:\report\quenched_disorder\report\fig\ave_peak20.png")
    plt.close()


if __name__ == "__main__":
    plot_peak(460, 2)
    # plot_rho_gas_vs_phi([440, 660, 880], [2, 3, 4])
