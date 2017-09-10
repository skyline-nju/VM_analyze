''' plot time-averaged density profile for epsilon=0.02 cases. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from axes_zoom_effect import zoom_effect03
from read_npz import read_matched_file, eq_Lx_and_nb


def plot_peak(Lx, nb, eta=350, eps=20):
    """ Plot time-averaged peaks for differernt samples with zoom effect."""

    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")

    fig = plt.figure(1, figsize=(8, 5))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(224)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(222)

    ax1.set_xlim(60, 210)
    ax3.set_xlim(190, 200)
    ax3.set_ylim(0.45, 0.49)
    zoom_effect03(ax3, ax1, 190, 200, loc="downward")

    dictLSN = read_matched_file({"Lx": Lx, "eta": eta, "eps": eps})
    phi = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dictLSN)])
    std_gap = np.array(
        [i for i in eq_Lx_and_nb(Lx, nb, "std_gap", dictLSN=dictLSN)])
    color_list = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])

    x = np.arange(Lx) + 0.5
    rho_gas = np.zeros_like(phi)

    for i, peak in enumerate(
            eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dictLSN)):
        ax1.plot(x, peak, c=color_list[i], lw=0.8)
        ax3.plot(x, peak, c=color_list[i])
        rho_gas[i] = np.mean(peak[190:200])
    ax2.axis("auto")
    ax4.axis("auto")
    sca = ax2.scatter(rho_gas, phi, c=phi, cmap="jet")
    ax4.scatter(std_gap, phi, c=phi, cmap="jet")
    z = np.polyfit(rho_gas, phi, 1)
    print(z)

    bbox = dict(edgecolor="k", fill=False)
    ax1.text(0.91, 0.91, "(a)", transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.91, 0.91, "(d)", transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.91, 0.91, "(c)", transform=ax3.transAxes, bbox=bbox)
    ax4.text(0.91, 0.91, "(b)", transform=ax4.transAxes, bbox=bbox)

    ylabel = r"$\langle \overline{\rho}_y (x)\rangle_t$"
    xlabel = r"$x$"
    ax1.text(0.01, 0.92, ylabel, transform=ax1.transAxes)
    ax1.text(0.96, 0.07, xlabel, transform=ax1.transAxes)
    ax2.text(0.01, 0.92, r"$\langle \phi \rangle_t$", transform=ax2.transAxes)
    ax2.text(0.90, 0.02, r"$\rho_{\rm gas}$", transform=ax2.transAxes)
    ax3.text(0.01, 0.92, ylabel, transform=ax3.transAxes)
    ax3.text(0.96, 0.02, xlabel, transform=ax3.transAxes)
    ax4.text(0.01, 0.92, r"$\langle \phi \rangle_t$", transform=ax4.transAxes)
    ax4.text(
        0.78,
        0.02,
        r"$\langle \sigma(\Delta x_b)\rangle_t$",
        transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\eta=%g,\epsilon=%g, \rho_0=1, L_x=%d, L_y=200, n_b=%d$" %
                 (eta / 1000, eps / 1000, Lx, nb))
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.08, 0.03, 0.8])
    fig.colorbar(sca, cax=cbar_ax)
    fig.text(0.88, 0.9, r"$\langle \phi \rangle_t$", color="b", fontsize=14)

    plt.show()
    # plt.savefig(
    #     r"E:\report\quenched_disorder\report\fig\ave_peak20.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_peak(460, 2)
