import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import common


def plot_serials(eta, eps):
    """ Plot time serials of number of bands and order parameters. """

    import handle

    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = common.get_para(file)
        outfile = "ts_%d.%d.%d.%d.%d.png" % (para[0], para[1], para[2],
                                             para[3], para[4])
        t_beg, t_end = bf["t_beg_end"]
        handle.plot_serials(
            para,
            t_beg,
            t_end,
            bf["num_raw"],
            bf["num_smoothed"],
            bf["seg_num"],
            bf["seg_idx0"],
            bf["seg_idx1"],
            bf["seg_phi"],
            bf["beg_movAve"],
            bf["end_movAve"],
            bf["phi_movAve"],
            outfile=outfile)


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
            para = []
        dict_LSN = get_dict_Lx_seed_nb(para)
    dict_NLS = {Lx: {} for Lx in dict_LSN}
    for Lx in dict_LSN:
        dict_NLS[Lx] = common.swap_key(dict_LSN[Lx])
    dict_NLS = common.swap_key(dict_NLS)
    return dict_NLS


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
        if para is None:
            para = ["Lx", Lx]
        dict_NLS = get_dict_nb_Lx_seed(para, dict_LSN)

    dict_S = dict_NLS[nb][Lx]
    phi = []
    std_gap = []
    rate = []
    for seed in dict_S:
        phi.append(dict_S[seed]["mean_phi"])
        std_gap.append(dict_S[seed]["std_gap"])
        rate.append(dict_S[seed]["rate"])

    print("max rate = %g, min rate = %g" % (max(rate), min(rate)))
    flag_show = False
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    sca = ax.scatter(std_gap, phi, c=rate)
    if flag_show:
        plt.colorbar(sca)
        ax.set_xlabel(r"$\Delta n_b$")
        ax.set_ylabel(r"$\langle\phi\rangle_t$")
        plt.show()
        plt.close()


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


def get_peaks(nb, eta, eps, Lx, Ly=200, dict_LSN=None):
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
    rho_gas = np.array([np.mean(peak[190:195]) for peak in peak_list])
    return peak_list, phi_list, rho_gas


def plot_peak_varied_sample(nb,
                            eta,
                            eps,
                            Lx,
                            Ly=200,
                            dict_LSN=None,
                            ax=None,
                            ax_phi=None):
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

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\overline{\rho}_y(x)$")

    if flag_show:
        plt.axvline(180)
        plt.axhline(1.8)
        ax.set_title(
            r"$\eta=%g, \epsilon=%g, \rho_0=1, L_x=%d, L_y=%d, n_b=%d$" %
            (eta / 1000, eps / 1000, Lx, Ly, nb))
        plt.show()
        plt.close()

    if ax_phi is not None:
        ax_phi.axis("auto")
        rho_gas = [np.mean(peak[190:195]) for peak in peak_list]
        sca = ax_phi.scatter(rho_gas, phi_list, c=phi_list, cmap="jet")
        ax_phi.set_xlabel(r"$\rho_{\rm{gas}}$")
        ax_phi.set_ylabel(r"$\langle \phi \rangle_t$")
        return sca


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")
    # phi_nb1_vs_phi_nb2(int(sys.argv[1]))
    # phi_vs_std_gap(2, int(sys.argv[1]))
    # plot_peak_varied_sample(2, 350, 20, int(sys.argv[1]))
