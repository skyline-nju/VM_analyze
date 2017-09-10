import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import common


def get_dict_nb(buff, rate_min=0.3) -> dict:
    """ Get a dict with key nb from data buff

        Parameters:
        --------
            buff: dict
                Data read from npz file.
            rate_min: float
                Minimal rate to accept.

        Returns:
        --------
            res: dict
                {nb:{"mean_phi": mean_phi, "rate": rate}}
    """
    sum_phi = {}
    count = {}
    nbs = buff["seg_num"]
    phis = buff["seg_phi"]
    lens = buff["seg_idx1"] - buff["seg_idx0"]
    tot = buff["seg_idx1"][-1] - buff["seg_idx0"][0]
    for i, nb in enumerate(nbs):
        if nb > 0:
            if nb in sum_phi:
                sum_phi[nb] += phis[i] * lens[i]
                count[nb] += lens[i]
            else:
                sum_phi[nb] = phis[i] * lens[i]
                count[nb] = lens[i]
        # tot += lens[i]
    res = {}
    for i, nb in enumerate(buff["num_set"]):
        if nb > 0:
            rate = count[nb] / tot
            if rate > rate_min:
                if nb not in res:
                    res[nb] = {}
                res[nb]["mean_phi"] = sum_phi[nb] / count[nb]
                res[nb]["rate"] = rate
                res[nb]["std_gap"] = buff["sum_std_gap"][i] / buff[
                    "count_rhox"][i]
                res[nb]["ave_peak"] = buff["sum_rhox"][i] / buff["count_rhox"][
                    i]
    return res


def get_dict_Lx_seed_nb(para: list=None) -> dict:
    """ Get dict with key: Lx->Seed->nb. """
    if para is None:
        para = []
    pat = common.list2str(para, "eta", "eps", "Lx", "Ly", "seed")
    files = glob.glob("mb_%s.npz" % (pat))
    res = {}
    for file in files:
        buff = np.load(file)
        para = common.get_para(file)
        Lx = para[2]
        seed = para[4]
        if Lx not in res:
            res[Lx] = {seed: {}}
        elif Lx in res and seed not in res[Lx]:
            res[Lx][seed] = {}
        res[Lx][seed] = get_dict_nb(buff)
    return res


def phi_nb1_vs_phi_nb2(Lx,
                       nb1=None,
                       nb2=None,
                       para=None,
                       dict_LSN=None,
                       ax=None):
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
        if nb1 is None:
            nb1 = Lx // 220
        nb2 = nb1 + 1
    if dict_LSN is None:
        if para is None:
            para = ["Lx", Lx]
        dict_LSN = get_dict_Lx_seed_nb(para)
    dict_SN = dict_LSN[Lx]

    phi_nb1 = []
    phi_nb2 = []
    rate = []
    for seed in dict_SN:
        if nb1 in dict_SN[seed] and nb2 in dict_SN[seed]:
            phi_nb1.append(dict_SN[seed][nb1]["mean_phi"])
            phi_nb2.append(dict_SN[seed][nb2]["mean_phi"])
            rate.append(dict_SN[seed][nb2]["rate"] /
                        dict_SN[seed][nb1]["rate"])

    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True

    phi_nb1 = np.array(phi_nb1)
    phi_nb2 = np.array(phi_nb2)
    sca = ax.scatter(phi_nb1, phi_nb2 - phi_nb1, c=rate, cmap="jet")

    if flag_show:
        plt.colorbar(sca)
        ax.set_xlabel(r"$\langle \phi \rangle_t(n_b=%d)$" % nb1)
        ax.set_ylabel(r"$\langle \phi \rangle_t(n_b=%d)$" % nb2)
        ax.set_title(r"$L_x=%d$" % (Lx))
        plt.show()
        plt.close()

    for i in range(len(phi_nb1)):
        print("%f\t%f\t%f" % (phi_nb1[i], phi_nb2[i], rate[i]))


def get_rho_gas(nb, eta, eps, Lx, Ly=200, dict_LSN=None):
    """ Get list of peak, phi and rho_gas. """

    if dict_LSN is None:
        para = ["eta", str(eta), "eps", str(eps), "Lx", str(Lx), "Ly", str(Ly)]
        dict_LSN = get_dict_Lx_seed_nb(para)
    dict_SN = dict_LSN[Lx]

    peak_list = []
    phi_list = []
    for seed in dict_SN:
        if nb in dict_SN[seed]:
            peak_list.append(dict_SN[seed][nb]["ave_peak"])
            phi_list.append(dict_SN[seed][nb]["mean_phi"])

    phi = np.array(phi_list)
    rho_gas = np.array([np.mean(peak[190:195]) for peak in peak_list])
    return rho_gas, phi


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")
    # phi_nb1_vs_phi_nb2(int(sys.argv[1]))
    # phi_vs_std_gap(2, int(sys.argv[1]))
    # plot_peak_varied_sample(2, 350, 20, int(sys.argv[1]))
