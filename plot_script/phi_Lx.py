""" Plot phi against Lx. """

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from axes_zoom_effect import zoom_effect03
sys.path.append("../")

try:
    import ana_data
except:
    raise


def phi_Lx_const_nb(nb,
                    para=None,
                    dict_LSN=None,
                    dict_NLS=None,
                    fit=None,
                    vlim=None,
                    ax=None):
    """ Plot phi against Lx at given nb.

        Parameters:
        --------
            nb: int
                Number of bands.
            Para: list
                List of parameters: eta, $eta, eps, $eps...
            dict_LSN: dict
                A dict with keys: Lx->seed->nb
            dict_NLS: dict
                A dict with keys: nb->Lx->seed
            fit: str
                Method to fit the plot of sample-averaged order parameters.
            vlim: list
                The limitation of color list.
            ax: matplotlib.axes
                Axes of matplotlib

        Returns:
        --------
            sca: plt.scatter
                Scatter plot.
    """
    if dict_NLS is None:
        dict_NLS = ana_data.get_dict_nb_Lx_seed(para, dict_LSN)
    dict_LS = dict_NLS[nb]
    phi_dict = {Lx: [] for Lx in dict_LS}
    rate_dict = {Lx: [] for Lx in dict_LS}
    for Lx in dict_LS:
        for seed in dict_LS[Lx]:
            rate = dict_LS[Lx][seed]["rate"]
            phi = dict_LS[Lx][seed]["mean_phi"]
            phi_dict[Lx].append(phi)
            rate_dict[Lx].append(rate)

    if vlim is not None:
        vmin, vmax = vlim
    else:
        rate_all = []
        for Lx in rate_dict:
            rate_all += rate_dict[Lx]
        vmin = min(rate_all)
        vmax = 1

    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    for Lx in phi_dict:
        phi_dict[Lx] = np.array(phi_dict[Lx])
        rate_dict[Lx] = np.array(rate_dict[Lx])
        for i, phi in enumerate(phi_dict[Lx]):
            sca = ax.scatter(Lx, phi, c=rate_dict[Lx][i], vmin=vmin, vmax=vmax)
    Lxs = np.array(sorted(dict_LS.keys()))
    for Lx in Lxs:
        print("%d: min=%g, max=%g" %
              (Lx, rate_dict[Lx].min(), rate_dict[Lx].max()))
    mean_phi = np.array([phi_dict[Lx].mean() for Lx in Lxs])
    std_phi = np.array([phi_dict[Lx].std() for Lx in Lxs])

    if fit == "linear":
        ax.errorbar(Lxs, mean_phi, std_phi, fmt="rs")
        z = np.polyfit(Lxs, mean_phi, 1)
        print("z = ", z)
        x = np.linspace(Lxs[0]-10, Lxs[-1]+10, 50)
        y = z[1] + z[0]*x
        ax.plot(x, y, "--r")
    else:
        ax.errorbar(Lxs, mean_phi, std_phi, fmt="--rs")

    if flag_show:
        cbar = plt.colorbar(sca, ax=ax)
        cbar.set_label("Probability")
        ax.set_xlabel(r"$L_x$")
        ax.set_ylabel(r"$\langle \phi \rangle_t$")

        plt.title(r"$\eta=0.35, \epsilon=0.02, \rho_0=1, L_y=200, n_b=%d$" %
                  (nb))
        plt.tight_layout()
        plt.show()
        plt.close()
    elif vlim is None:
        vlim = [vmin, vmax]
        return sca, vlim


def two_panel():
    nb = 2
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

    ymin = 0.427
    ymax = 0.452
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps20")
    sca, vlim = phi_Lx_const_nb(nb, ax=ax1)
    ax1.set_ylim(0.425, 0.4565)

    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")
    phi_Lx_const_nb(nb, ax=ax2, vlim=vlim, fit="linear")
    ax2.set_ylim(ymin, ymax)

    zoom_effect03(ax2, ax1, 390, 490, ymin=ymin, ymax=ymax, loc="rightward")

    # set label and title
    ax1.set_xlabel(r"$L_x$")
    ax2.set_xlabel(r"$L_x$")
    ax1.set_ylabel(r"$\langle \phi \rangle_t$")
    ax1.set_title(r"(a) several samples per $L_x$")
    ax2.set_title(r"(b) around 50 samples per $L_x$")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(
        r"$\eta=0.35,\epsilon=0.02, \rho_0=1, L_y=200, n_b=%d$" % (nb),
        color="b")

    # add colorbar
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
    cb = fig.colorbar(sca, cax=cbar_ax)
    cb.set_label("Probability")

    # plt.show()
    plt.savefig(r"E:\report\quenched_disorder\report\fig\phi_Lx_zoom.png")
    plt.close()


if __name__ == "__main__":
    two_panel()
