import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multBand as mb
import analyze_res as ana
import glob


def get_dict_nb(nbs: list, phis: list, lens: list) -> dict:
    """ Transform array of phi into a dict with key nb.

        Parameters:
        --------
            nbs: list
                List of number of bands, as keys of dict0
            phis: list
                List of order parameters of each segements.
            lens: list
                Lengths of each segements.

        Returns:
        --------
            res: dict
                {nb:{"mean_phi": mean_phi, "rate": rate}}
    """
    sum_phi = {}
    count = {}
    tot = 0
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
    for nb in res:
        res[nb]["mean_phi"] = sum_phi[nb] / count[nb]
        res[nb]["rate"] = count[nb] / tot
    return res


def get_dict_Lx_seed_nb(para: list) -> dict:
    """ Get dict with key: Lx->Seed->nb. """
    pat = mb.list2str(para, "eta", "eps", "Lx", "Ly", "seed")
    files = glob.glob("mb_%s.npz" % (pat))
    res = {}
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        seed = para[4]
        if Lx not in res:
            res[Lx] = {seed: {}}
        elif Lx in res and seed not in res[Lx]:
            res[Lx][seed] = {}
        res[Lx][seed] = get_dict_nb(bf["seg_num"], bf["seg_phi"],
                                    bf["seg_idx1"] - bf["seg_idx0"])
    return res


def get_dict_nb_Lx_seed(dict0: dict) -> dict:
    """ Get dict with key: nb->Lx->seed. """
    res = {Lx: {} for Lx in dict0}
    for Lx in dict0:
        res[Lx] = ana.swap_key(dict0[Lx])
    res = ana.swap_key(res)
    return res


def plot_phi_L(para: list, nb):
    """ Plot phi vs Lx. """
    dict0 = get_dict_Lx_seed_nb(para)
    dict1 = get_dict_nb_Lx_seed(dict0)
    dict2 = dict1[nb]
    phi_dict = {Lx: [] for Lx in dict2}
    for Lx in dict2:
        for seed in dict2[Lx]:
            rate = dict2[Lx][seed]["rate"]
            phi = dict2[Lx][seed]["mean_phi"]
            if rate > 0.3:
                phi_dict[Lx].append(phi)
    for Lx in phi_dict:
        phi_dict[Lx] = np.array(phi_dict[Lx])
        plt.plot([Lx] * phi_dict[Lx].size, phi_dict[Lx], "ko", ms=2)
    Lxs = sorted(dict2.keys())
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


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\tmp")
    plot_phi_L(sys.argv[1:], 2)
