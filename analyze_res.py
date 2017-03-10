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


def get_dict_sum(seg_nb: np.ndarray,
                 seg_len: np.ndarray,
                 seg_phi: np.ndarray,
                 dict0: dict):
    """ Transform array of phi into dict type with key nb.

        Parameters:
        --------
            seg_nb: np.ndarray
                Number of bands for each segment.
            seg_len: np.ndarray
                Length of each segment of time serials.
            seg_phi: np.ndarray
                Mean order parameter of each segment.
            dict0: dict
                Dict as input and output
    """
    for i, nb in enumerate(seg_nb):
        if nb > 0:
            sum_phi = seg_len[i] * seg_phi[i]
            count = seg_len[i]
            if nb in dict0:
                dict0[nb]["sum_phi"] += sum_phi
                dict0[nb]["count_phi"] += count
            else:
                dict0[nb] = {"sum_phi": sum_phi, "count_phi": count}


def get_dict_mean(dict0: dict, layer: int) -> dict:
    """ Get time (and sample) average of phi.

        Parameters:
        --------
            dict0: dict
                Dually or triply nested
            level: int
                Layer of dict0.
        Returns:
        --------
            dict1: dict
                Averaged phi and corresponding ratio.
    """
    dict1 = {}
    if layer == 2:
        tot = sum([dict0[nb]["count_phi"] for nb in dict0])
        for nb in dict0:
            dict1[nb] = {nb: {"mean_phi": 0, "rate_phi": 0}}
            dict1[nb]["mean_phi"] = dict0[nb]["sum_phi"] / dict0[nb][
                "count_phi"]
            dict1[nb]["rate_phi"] = dict0[nb]["count_phi"] / tot
    elif layer == 3:
        tot = 0  # key = nb
        dict1 = {}  # key = nb
        for seed in dict0:
            for nb in dict0[seed]:
                if nb in tot:
                    dict1[nb]["mean_phi"] += dict0[seed][nb]["sum_phi"]
                    dict1[nb]["rate_phi"] += dict0[seed][nb]["count_phi"]
                else:
                    dict1[nb]["mean_phi"] = dict0[seed][nb]["sum_phi"]
                    dict1[nb]["rate_phi"] = dict0[seed][nb]["count_phi"]
                tot += dict0[seed][nb]["count_phi"]
        for nb in tot:
            dict1[nb]["mean_phi"] /= tot
            dict1[nb]["rate_phi"] /= tot
    return dict1


def sum_over_time(para: list) -> dict:
    """ sum phi over time for each nb, respectively."""
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
        get_dict_sum(bf["seg_num"], bf["seg_idx1"] - bf["seg_idx0"],
                     bf["seg_phi"], res[Lx][seed])
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


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    # file = "mb_350.0.740.200.214740.npz"
    # buff = np.load(file)
    # para = mb.get_para(file)
    # mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"],
    #                   buff["count_rhox"])
    # plot_serials(350, 20)
    sum_t = sum_over_time(sys.argv[1:])
    plot_time_ave_peak(sum_t, 360)
