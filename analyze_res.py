import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import multBand as mb


def plot_serials(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        outfile = "ts_%d.%d.%d.%d.%d.png" % (para[0], para[1], para[2],
                                             para[3], para[4])
        t_beg, t_end = bf["t_beg_end"]
        mb.plot_serials(
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


def plot_peak(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        mb.plot_rhox_mean(para, bf["num_set"], bf["sum_rhox"],
                          bf["count_rhox"])


def sum_over_time(para: list) -> dict:
    """ Genereate a dict with keys following the order of
        Lx->seed->nb->(sum_phi, count_phi, sum_rhox, count_rhox).

        Parameters:
        --------
            para: list
                List of parameters such as "eta", eta, "eps", eps...

        Returns:
        --------
            res: list
                A dict with form {Lx:{seed:{nb:{"sum_phi": sum_phi,...}}}}
    """

    def list2dict(dict0: dict, nbs: list, phis: list, lens: list):
        """ Transform an array into a dict.

            Parameters:
            --------
                dict0: dict
                    Input dict, also as output.
                nbs: list
                    List of number of bands, as keys of dict0
                phis: list
                    List of order parameters of each segements.
                lens: list
                    Lengths of each segements.
        """
        for i, nb in enumerate(nbs):
            if nb > 0:
                sum_phi = lens[i] * phis[i]
                if nb in dict0:
                    dict0[nb]["sum_phi"] += sum_phi
                    dict0[nb]["count_phi"] += lens[i]
                else:
                    dict0[nb] = {"sum_phi": sum_phi, "count_phi": lens[i]}

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
        list2dict(res[Lx][seed], bf["seg_num"], bf["seg_phi"],
                  bf["seg_idx1"] - bf["seg_idx0"])
        for i, nb in enumerate(bf["num_set"]):
            if nb > 0:
                res[Lx][seed][nb]["sum_rhox"] = bf["sum_rhox"][i]
                res[Lx][seed][nb]["count_rhox"] = bf["count_rhox"][i]
    return res


def time_ave(dict0: dict) -> dict:
    """ Time average for phi and peak.

        Parameters:
        --------
            dict0: dict
                Return of fun: sum_over_time

        Returns:
        --------
            res: dict
                Time-averaged phi, rho_x and coresponding rate.
    """
    res = {}
    for Lx in dict0:
        res[Lx] = {}
        for seed in dict0[Lx]:
            res[Lx][seed] = {}
            tot = 0
            for nb in dict0[Lx][seed]:
                res[Lx][seed][nb] = {}
                res[Lx][seed][nb]["mean_phi"] = dict0[Lx][seed][nb][
                    "sum_phi"] / dict0[Lx][seed][nb]["count_phi"]
                res[Lx][seed][nb]["mean_rhox"] = dict0[Lx][seed][nb][
                    "sum_rhox"] / dict0[Lx][seed][nb]["count_rhox"]
                tot += dict0[Lx][seed][nb]["count_phi"]
            for nb in dict0[Lx][seed]:
                res[Lx][seed][nb]["rate"] = dict0[Lx][seed][nb][
                    "count_phi"] / tot
    return res


def sample_ave(dict0: dict) -> dict:
    """ Sample average for phi and peak.

        Parameters:
        --------
            dict0: dict
                Return of fun: sum_over_time

        Returns:
        --------
            res: dict
                Sample-averaged phi, rho_x and coresponding rate.
    """
    res = {}
    for Lx in dict0:
        res[Lx] = {}
        tot = 0
        for seed in dict0[Lx]:
            for nb in dict0[Lx][seed]:
                if nb in res[Lx]:
                    res[Lx][nb]["sum_phi"] += dict0[Lx][seed][nb]["sum_phi"]
                    res[Lx][nb]["count_phi"] += dict0[Lx][seed][nb][
                        "count_phi"]
                    res[Lx][nb]["sum_rhox"] += dict0[Lx][seed][nb]["sum_rhox"]
                    res[Lx][nb]["count_rhox"] += dict0[Lx][seed][nb][
                        "count_rhox"]
                else:
                    res[Lx][nb] = {}
                    res[Lx][nb]["sum_phi"] = dict0[Lx][seed][nb]["sum_phi"]
                    res[Lx][nb]["count_phi"] = dict0[Lx][seed][nb]["count_phi"]
                    res[Lx][nb]["sum_rhox"] = dict0[Lx][seed][nb]["sum_rhox"]
                    res[Lx][nb]["count_rhox"] = dict0[Lx][seed][nb][
                        "count_rhox"]
                tot += dict0[Lx][seed][nb]["count_phi"]
        for nb in res[Lx]:
            res[Lx][nb]["mean_phi"] = res[Lx][nb]["sum_phi"] / res[Lx][nb][
                "count_phi"]
            res[Lx][nb]["rate"] = res[Lx][nb]["count_phi"] / tot
            res[Lx][nb]["mean_rhox"] = res[Lx][nb]["sum_rhox"] / res[Lx][nb][
                "count_rhox"]
    return res


def plot_phi(sum_t: dict, SampleAve=False, rate_m=0.2):
    if SampleAve is False:
        ave_t = time_ave(sum_t)
        for Lx in ave_t:
            for seed in ave_t[Lx]:
                for nb in ave_t[Lx][seed]:
                    rate = ave_t[Lx][seed][nb]["rate"]
                    if rate > rate_m:
                        phi = ave_t[Lx][seed][nb]["mean_phi"]
                        plt.scatter(Lx, phi, s=4, c=rate)
    else:
        ave_s = sample_ave(sum_t)
        for Lx in ave_s:
            for nb in ave_s[Lx]:
                rate = ave_s[Lx][nb]["rate"]
                if rate > rate_m:
                    phi = ave_s[Lx][nb]["mean_phi"]
                    plt.scatter(Lx, phi, s=4, c=rate)
    plt.colorbar()
    plt.show()
    plt.close()


def plot_nb(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        for nb in bf["num_set"]:
            if nb > 0:
                plt.plot(1 / Lx, nb / Lx, "o")
    plt.show()
    plt.close()


def plot_time_ave_peak(sum_t, Lx):
    """ Plot time-averaged peak. """
    ave_t = time_ave(sum_t)
    data = swap_key(ave_t[Lx])
    x = np.arange(Lx) + 0.5
    for nb in data:
        for seed in data[nb]:
            plt.plot(
                x,
                data[nb][seed]["mean_rhox"],
                label=r"$\rm{seed}=%d, \phi=%f$" %
                (seed, data[nb][seed]["mean_phi"]))
        plt.legend(loc="best")
        plt.title(r"$L_x=%d, n_b=%d$" % (Lx, nb))
        plt.show()
        plt.close()


def swap_key(dict0):
    """ Swap keys of a nested dict.

        Parameters:
        --------
        dict0: dict
            A dict that is at least dually nested.

        Returns:
        dict1: dict
            By swaping keys in the first and second layer of dict0.
    """
    dict1 = {}
    for key1 in dict0:
        for key2 in dict0[key1]:
            if key2 in dict1:
                dict1[key2][key1] = dict0[key1][key2]
            else:
                dict1[key2] = {key1: dict0[key1][key2]}
    return dict1


def diff_peak(Lx, eta=350, Ly=200):
    """ Making a comparision betwwen time-averaged peaks of varied eps."""
    eps1 = 0
    eps2 = 20
    sum_t_1 = sum_over_time(["eta", eta, "eps", eps1, "Lx", Lx, "Ly", Ly])
    sum_t_2 = sum_over_time(["eta", eta, "eps", eps2, "Lx", Lx, "Ly", Ly])
    ave_t_1 = swap_key(time_ave(sum_t_1)[Lx])
    ave_t_2 = swap_key(time_ave(sum_t_2)[Lx])
    x = np.arange(Lx) + 0.5
    for nb in ave_t_1:
        if nb in ave_t_2:
            for sd1 in ave_t_1[nb]:
                plt.plot(
                    x,
                    ave_t_1[nb][sd1]["mean_rhox"],
                    "-",
                    label=r"$\epsilon=%g, \rm{seed}=%d, \phi=%f$" %
                    (eps1 / 1000, sd1, ave_t_1[nb][sd1]["mean_phi"]))
            for sd2 in ave_t_2[nb]:
                plt.plot(
                    x,
                    ave_t_2[nb][sd2]["mean_rhox"],
                    "--",
                    label=r"$\epsilon=%g, \rm{seed}=%d, \phi=%f$" %
                    (eps2 / 1000, sd2, ave_t_2[nb][sd2]["mean_phi"]))
            plt.legend(loc="best")
            plt.title(r"$L_x=%d, L_y=%d, n_b=%d$" % (Lx, Ly, nb))
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\overline{\rho}_y(x)$")
            plt.show()
            plt.close()


def phi_nb1_vs_phi_nb2(Lx, nb1, nb2, eta=350, eps=20, Ly=200):
    """"""
    sum_t = sum_over_time(["eta", eta, "eps", eps, "Lx", Lx, "Ly", Ly])
    ave_t = time_ave(sum_t)[Lx]
    for seed in ave_t:
        if nb1 in ave_t[seed] and nb2 in ave_t[seed]:
            plt.plot(
                ave_t[seed][nb1]["mean_phi"],
                ave_t[seed][nb2]["mean_phi"],
                "o",
                label="seed=%d" % (seed))
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    # file = "mb_350.0.740.200.214740.npz"
    # buff = np.load(file)
    # para = mb.get_para(file)
    # mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"],
    #                   buff["count_rhox"])
    # plot_serials(350, 20)
    # sum_t = sum_over_time(sys.argv[1:])
    # plot_time_ave_peak(sum_t, 360)
    # diff_peak(int(sys.argv[1]))
    phi_nb1_vs_phi_nb2(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[2])+1)
