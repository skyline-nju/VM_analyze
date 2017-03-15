# plot time-averaged density profile for epsilon=0 cases.

import numpy as np
import sys
import os
import matplotlib
from fractions import Fraction
sys.path.append("../")
# matplotlib.use("PS")
print(matplotlib.get_backend())
try:
    import ana_data
    import matplotlib.pyplot as plt
except:
    raise


def equal_Lr_over_nb(ratio: Fraction, ax: matplotlib.axes=None,
                     lambd: int=180):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    for Lx in dict_LSN:
        for seed in dict_LSN[Lx]:
            for nb in dict_LSN[Lx][seed]:
                Lr = Lx - nb * lambd
                if ratio * nb * lambd == Lr:
                    ax.plot(
                        np.arange(Lx) + 0.5,
                        dict_LSN[Lx][seed][nb]["ave_peak"],
                        label=r"$L_x=%d,n_b=%d$" % (Lx, nb),
                        lw=0.5)

    ax.set_xlim(80, 200)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\langle \overline{\rho}_y(x)\rangle_t$")
    ax.legend(title=r"$L_r/n_b\lambda=%s$" % str(ratio))
    if flag_show:
        plt.show()
        plt.close()


def equal_nb(nb0: int, ax: matplotlib.axes=None, lambd=180):
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
                    peaks.append(dict_LSN[Lx][seed][nb]["ave_peak"])
            break  # only show one sample

    clist = plt.cm.viridis(np.linspace(0, 1, len(Lxs)))
    for i, peak in enumerate(peaks):
        ax.plot(
            np.arange(Lxs[i]) + 0.5,
            peak,
            c=clist[i],
            lw=0.8,
            label=r"$L_x=%d,L_r/n_b\lambda=%s$" % (Lxs[i], fracs[i]))
    ax.set_xlim(80, 200)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\langle \overline{\rho}_y(x)\rangle_t$")
    ax.legend(title=r"$n_b=%d$" % nb0, loc="best", fontsize="x-small")
    if flag_show:
        plt.show()
        plt.close()


def eq_ratio3():
    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, nrows=1, figsize=(14, 5), sharey=True)
    equal_Lr_over_nb(Fraction(-1, 18), ax1)
    equal_Lr_over_nb(Fraction(0, 18), ax2)
    equal_Lr_over_nb(Fraction(1, 18), ax3)
    plt.tight_layout()
    # plt.suptitle(r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200$")
    plt.show()
    plt.close()


def eq_nb3():
    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, nrows=1, figsize=(14, 5), sharey=True)
    equal_nb(2, ax1)
    equal_nb(3, ax2)
    equal_nb(4, ax3)
    plt.tight_layout()
    # plt.suptitle(r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200$")
    plt.show()
    plt.close()


def four_panel():
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 7))
    equal_Lr_over_nb(Fraction(0, 18), axes[0][0])
    equal_Lr_over_nb(Fraction(-1, 18), axes[0][1])
    equal_Lr_over_nb(Fraction(1, 18), axes[1][0])
    equal_nb(2, axes[1][1])
    plt.suptitle(r"$\eta=0.35,\epsilon=0, \rho_0=1, L_y=200$")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    plt.savefig(r"E:\report\quenched_disorder\report\figave_peak0.eps")
    plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps0")
    dict_LSN = ana_data.get_dict_Lx_seed_nb()
    # equal_Lr_over_nb(Fraction(sys.argv[1]))
    # equal_nb(2)
    four_panel()
