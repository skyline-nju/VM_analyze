import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multBand as mb
import glob


def get_dict_nb(buff) -> dict:
    """ Get a dict with key nb from data buff

        Parameters:
        --------
            buff: dict
                Data read from npz file.

        Returns:
        --------
            res: dict
                {nb:{"mean_phi": mean_phi, "rate": rate}}
    """
    sum_phi = {}
    count = {}
    tot = 0
    nbs = buff["seg_num"]
    phis = buff["seg_phi"]
    lens = buff["seg_idx1"] - buff["seg_idx0"]
    for i, nb in enumerate(nbs):
        if nb > 0:
            if nb in sum_phi:
                sum_phi[nb] += phis[i] * lens[i]
                count[nb] += lens[i]
            else:
                sum_phi[nb] = phis[i] * lens[i]
                count[nb] = lens[i]
        tot += lens[i]
    res = {nb: {} for nb in sum_phi}
    for i, nb in enumerate(buff["num_set"]):
        if nb > 0:
            res[nb]["mean_phi"] = sum_phi[nb] / count[nb]
            res[nb]["rate"] = count[nb] / tot
            res[nb]["std_gap"] = buff["sum_std_gap"][i] / buff["count_rhox"][
                i] * nb
            res[nb]["ave_peak"] = buff["sum_rhox"][i] / buff["count_rhox"][i]
    return res


def get_dict_Lx_seed_nb(para: list) -> dict:
    """ Get dict with key: Lx->Seed->nb. """
    pat = mb.list2str(para, "eta", "eps", "Lx", "Ly", "seed")
    files = glob.glob("mb_%s.npz" % (pat))
    res = {}
    for file in files:
        buff = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        seed = para[4]
        if Lx not in res:
            res[Lx] = {seed: {}}
        elif Lx in res and seed not in res[Lx]:
            res[Lx][seed] = {}
        res[Lx][seed] = get_dict_nb(buff)
    return res


def get_dict_nb_Lx_seed(para=None, dict_LSN=None) -> dict:
    """ Get dict with key: nb->Lx->seed.

        Parameters:
        --------
            para: list
                List of parameters: eta, $eta, eps, $eps...
            dict_LSN: dict
                 A dict with keys: Lx->seed->nb

        Returns:
        --------
            dict_NLS: dict
                A dict with keys: nb->Lx->seed
    """
    if dict_LSN is None:
        if para is None:
            print("Error, no input data!")
            sys.exit()
        else:
            dict_LSN = get_dict_Lx_seed_nb(para)
    dict_NLS = {Lx: {} for Lx in dict_LSN}
    for Lx in dict_LSN:
        dict_NLS[Lx] = mb.swap_key(dict_LSN[Lx])
    dict_NLS = mb.swap_key(dict_NLS)
    return dict_NLS


def phi_vs_Lx(nb, para=None, dict_LSN=None, dict_NLS=None, ax=None):
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
            ax: matplotlib.axes
                Axes of matplotlib
    """
    if dict_NLS is None:
        dict_NLS = get_dict_nb_Lx_seed(para, dict_LSN)
    dict_LS = dict_NLS[nb]
    phi_dict = {Lx: [] for Lx in dict_LS}
    for Lx in dict_LS:
        for seed in dict_LS[Lx]:
            rate = dict_LS[Lx][seed]["rate"]
            phi = dict_LS[Lx][seed]["mean_phi"]
            if rate > 0.3:
                phi_dict[Lx].append(phi)
    for Lx in phi_dict:
        phi_dict[Lx] = np.array(phi_dict[Lx])
        plt.plot([Lx] * phi_dict[Lx].size, phi_dict[Lx], "ko", ms=2)
    Lxs = sorted(dict_LS.keys())
    mean_phi = [phi_dict[Lx].mean() for Lx in Lxs]
    std_phi = [phi_dict[Lx].std() for Lx in Lxs]
    # plt.plot(Lxs, mean_phi, "s")
    plt.errorbar(Lxs, mean_phi, std_phi, fmt="--rs")
    plt.xlabel(r"$L_x$")
    plt.ylabel(r"$\langle \phi \rangle_t$")
    plt.title(r"$\eta=0.35, \epsilon=0.02, \rho_0=1, L_y=200$")
    plt.tight_layout()
    plt.show()
    plt.close()


def phi_vs_std_gap(nb, Lx, para=None, dict_LSN=None, dict_NLS=None, ax=None):
    """ Plot phi against std of gaps between nearest bands at given nb and Lx.

        Parameters:
        --------
            nb: int
                Number of bands.
            Lx: int
                Box size in x direction.
            para: list
                List of parameters: eta, $eta, eps, $eps...
            dict_LSN: dict
                A dict with keys: Lx->seed->nb
            dict_NLS: dict
                A dict with keys: nb->Lx->seed
            ax: matplotlib.axes
                Axes of matplotlib
    """
    if dict_NLS is None:
        dict_NLS = get_dict_nb_Lx_seed(para, dict_LSN)

    dict_S = dict_NLS[nb][Lx]
    phi = []
    std_gap = []
    for seed in dict_S:
        phi.append(dict_S[seed]["mean_phi"])
        std_gap.append(dict_S[seed]["std_gap"])
    flag_show = False
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    ax.plot(std_gap, phi, "o", label=r"$L_x=%d, n_b=%d$" % (Lx, nb))
    if flag_show:
        ax.set_xlabel(r"$\Delta n_b$")
        ax.set_ylabel(r"$\langle\phi\rangle_t$")
        plt.show()
        plt.close()


def phi_nb1_vs_phi_nb2(Lx, nb1, nb2=None, para=None, dict_LSN=None, ax=None):
    """ Plot phi(nb1) vs phi(nb2) at given Lx.

        Parameters:
        --------
            Lx: int
                Box size in x direction.
            nb1: int
                Number of bands at Lx.
            nb2: int
                Number of bands at Lx.
            para: list
                List of parameters: eta, $eta, eps, $eps...
            dict_LSN: dict
                A dict with keys: Lx->seed->nb
            ax: matplotlib.axes
                Axes of matplotlib
    """
    if nb2 is None:
        nb2 = nb1 + 1
    if dict_LSN is None:
        dict_LSN = get_dict_Lx_seed_nb(para)
    dict_SN = dict_LSN[Lx]

    phi_nb1 = []
    phi_nb2 = []
    for seed in dict_SN:
        if nb1 in dict_SN[seed] and nb2 in dict_SN[seed]:
            phi_nb1.append(dict_SN[seed][nb1]["mean_phi"])
            phi_nb2.append(dict_SN[seed][nb2]["mean_phi"])

    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    ax.plot(phi_nb1, phi_nb2, "o")

    if flag_show:
        ax.set_xlabel(r"$\langle \phi \rangle_t(n_b=%d)$" % nb1)
        ax.set_ylabel(r"$\langle \phi \rangle_t(n_b=%d)$" % nb2)
        plt.show()
        plt.close()


def plot_peak_varied_sample(nb, eta, eps, Lx, Ly=200, dict_LSN=None, ax=None):
    """ Plot peaks of differernt samples at given nb, eta, eps, Lx, Ly. """

    if dict_LSN is None:
        para = ["eta", str(eta), "eps", str(eps), "Lx", str(Lx), "Ly", str(Ly)]
        dict_LSN = get_dict_Lx_seed_nb(para)
    dict_SN = dict_LSN[Lx]

    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    peak_list = []
    phi_list = []
    for seed in dict_SN:
        if nb in dict_SN[seed]:
            peak_list.append(dict_SN[seed][nb]["ave_peak"])
            phi_list.append(dict_SN[seed][nb]["mean_phi"])

    phi_m = min(phi_list)
    d_phi = max(phi_list) - phi_m
    color_list = plt.cm.jet([(phi - phi_m) / d_phi for phi in phi_list])

    x = np.arange(Lx) + 0.5
    for i, peak in enumerate(peak_list):
        ax.plot(x, peak, c=color_list[i])

    if flag_show:
        ax.set_title(
            r"$\eta=%g, \epsilon=%g, \rho_0=1, L_x=%d, L_y=%d, n_b=%d$" %
            (eta / 1000, eps / 1000, Lx, Ly, nb))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\overline{\rho}_y(x)$")
        plt.show()
        plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\tmp")
    # phi_vs_Lx(2, para=sys.argv[1:])
    # phi_vs_std_gap(2, 480, para=sys.argv[1:])
    # phi_nb1_vs_phi_nb2(980, 4, para=sys.argv[1:])
    plot_peak_varied_sample(2, 350, 20, 420)
