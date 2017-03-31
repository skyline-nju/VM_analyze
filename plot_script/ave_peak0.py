''' plot time-averaged density profile for epsilon=0 cases. '''

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from fractions import Fraction
from read_npz import read_matched_file, fixed_para


def plot_eq_Lr_over_nb_lamb(ratio: Fraction,
                            ax: matplotlib.axes=None,
                            lamb: int=180):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    for peak, phi, Lx, nb in fixed_para(
            "ave_peak",
            "mean_phi",
            lamb=lamb,
            Lr_over_nb_lambda=ratio,
            dictLSN=dict_LSN):
        x = np.arange(Lx) + 0.5
        ax.plot(x, peak, label=r"$L_x=%d,n_b=%d$" % (Lx, nb), lw=0.5)

    ax.set_xlim(80, 200)
    ax.set_ylim(0, 4.5)
    ylabel = r"$\langle \overline{\rho}_y(x)\rangle_t$"
    ax.text(0.02, 0.92, ylabel, transform=ax.transAxes, fontsize="xx-large")
    ax.text(0.95, 0.02, r"$x$", transform=ax.transAxes, fontsize="xx-large")

    if flag_show:
        ax.legend(loc=(0.02, 0.6), fontsize="large", labelspacing=0.1)
        plt.show()
        plt.close()


def plot_eq_nb(nb0: int, ax: matplotlib.axes=None, lambd=180):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    peaks = []
    Lxs = []
    fracs = []
    for Lx in sorted(dict_LSN.keys()):
        for seed in dict_LSN[Lx]:
            for nb in dict_LSN[Lx][seed]:
                if nb == nb0:
                    fracs.append(Fraction(Lx - nb * lambd, nb * lambd))
                    Lxs.append(Lx)
                    peak = dict_LSN[Lx][seed][nb]["ave_peak"]
                    peaks.append(peak)
            break  # only show one sample

    clist = plt.cm.viridis(np.linspace(0, 1, len(Lxs)))
    for i, peak in enumerate(peaks):
        ax.plot(
            np.arange(Lxs[i]) + 0.5,
            peak,
            c=clist[i],
            lw=0.8,
            label=r"$L_r/n_b\lambda=%s$" % (fracs[i]))
    ax.set_xlim(80, 200)
    ax.set_ylim(0, 4.5)
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$\langle \overline{\rho}_y(x)\rangle_t$")
    ylabel = r"$\langle \overline{\rho}_y(x)\rangle_t$"
    ax.text(0.02, 0.92, ylabel, transform=ax.transAxes, fontsize="xx-large")
    ax.text(0.95, 0.02, r"$x$", transform=ax.transAxes, fontsize="xx-large")
    # ax.legend(
    #     title=r"$n_b=%d$" % nb0,
    #     loc=(0.02, 0.3),
    #     labelspacing=0)
    ax.set_title(r"$n_b=%d$" % nb0)
    if flag_show:
        plt.show()
        plt.close()


def eq_ratio3():
    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, nrows=1, figsize=(14, 5), sharey=True)
    plot_eq_Lr_over_nb_lamb(Fraction(-1, 18), ax1)
    plot_eq_Lr_over_nb_lamb(Fraction(0, 18), ax2)
    plot_eq_Lr_over_nb_lamb(Fraction(1, 18), ax3)
    plt.tight_layout()
    # plt.suptitle(r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200$")
    plt.show()
    plt.close()


def eq_nb3():
    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, nrows=1, figsize=(14, 5), sharey=True)
    plot_eq_nb(2, ax1)
    plot_eq_nb(3, ax2)
    plot_eq_nb(4, ax3)
    plt.tight_layout()
    # plt.suptitle(r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200$")
    plt.show()
    plt.close()


def four_panel():
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(14, 4), sharey=True)
    plot_eq_Lr_over_nb_lamb(Fraction(0, 18), axes[0])
    plot_eq_Lr_over_nb_lamb(Fraction(-1, 18), axes[1])
    plot_eq_Lr_over_nb_lamb(Fraction(1, 18), axes[2])
    plot_eq_nb(2, axes[3])

    axes[0].set_title(r"(a)$L_r/n_b\lambda=0$", fontsize="xx-large")
    axes[1].set_title(r"(b)$L_r/n_b\lambda=-1/18$", fontsize="xx-large")
    axes[2].set_title(r"(c)$L_r/n_b\lambda=1/18$", fontsize="xx-large")
    axes[3].set_title(r"(d)$n_b=2$", fontsize="xx-large")

    axes[3].arrow(
        0.5, 0.5, -0.3, 0.3, width=0.01, transform=axes[3].transAxes)
    axes[3].text(
        0.07,
        0.79,
        r"${\rm increasing}\ L_r/n_b\lambda$",
        transform=axes[3].transAxes,
        rotation=-45,
        fontsize="x-large")

    axes[0].legend(loc=(0.02, 0.6), fontsize=12.5, labelspacing=0.1)
    axes[1].legend(loc=(0.02, 0.7), fontsize=13, labelspacing=0.1)
    axes[2].legend(loc=(0.02, 0.7), fontsize=13, labelspacing=0.1)
    
    
    plt.suptitle(
        r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200, \lambda=180$",
        y=0.99,
        color="b",
        fontsize="xx-large")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # plt.savefig(r"E:\report\quenched_disorder\report\fig\ave_peak0.pdf")
    plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps0")
    dict_LSN = read_matched_file()
    # equal_Lr_over_nb(Fraction(sys.argv[1]))
    # equal_nb(2)
    four_panel()
