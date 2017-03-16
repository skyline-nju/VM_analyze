''' plot time-averaged density profile for epsilon=0.02 cases. '''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from axes_zoom_effect import zoom_effect02
sys.path.append("../")

try:
    import ana_data
except:
    raise


def plot_peaks():
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    nb = 2
    eta = 350
    eps = 20
    Lx = 460
    fig = plt.figure(1, figsize=(7, 7))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    ana_data.plot_peak_varied_sample(nb, eta, eps, Lx, ax=ax2)
    sca = ana_data.plot_peak_varied_sample(
        nb, eta, eps, Lx, ax=ax3, ax_phi=ax1)

    ax1.set_xlim(0.45, 0.49)
    ax1.set_ylim(0.42, 0.45)
    ax2.set_xlim(188, 196)
    ax2.set_ylim(0.45, 0.5)
    ax3.set_xlim(80, 200)
    zoom_effect02(ax2, ax3)

    ax1.set_title(r"$(a)$")
    ax2.set_title(r"$(b)$")
    ax3.set_title(r"$(c)$")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\eta=%g,\epsilon=%g, \rho_0=1, L_x=%d, L_y=200, n_b=%d$" %
                 (eta / 1000, eps / 1000, Lx, nb))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.11, 0.05, 0.75])
    cb = fig.colorbar(sca, cax=cbar_ax)
    cb.set_label(r"$\langle \phi \rangle_t$")

    # plt.show()
    plt.savefig(
        r"E:\report\quenched_disorder\report\fig\ave_peak20.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_peaks()
