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


def get_dict_nb_Lx_seed(dict0: dict) -> dict:
    """ Get dict with key: nb->Lx->seed. """
    res = {Lx: {} for Lx in dict0}
    for Lx in dict0:
        res[Lx] = mb.swap_key(dict0[Lx])
    res = mb.swap_key(res)
    return res


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
        if dict_LSN is None:
            if para is None:
                print("Error, no input data!")
                sys.exit()
            else:
                dict_LSN = get_dict_Lx_seed_nb(para)
        dict_NLS = get_dict_nb_Lx_seed(dict_LSN)
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
        if dict_LSN is None:
            if para is None:
                print("Error, no input data!")
                sys.exit()
            else:
                dict_LSN = get_dict_Lx_seed_nb(para)
        dict_NLS = get_dict_nb_Lx_seed(dict_LSN)

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


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\tmp")
    # phi_vs_Lx(2, para=sys.argv[1:])
    phi_vs_std_gap(2, 460, para=sys.argv[1:])
