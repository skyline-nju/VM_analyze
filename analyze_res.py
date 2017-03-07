import numpy as np
import os
import glob
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


def get_dict_sum(seg_nb: np.ndarray, seg_len: np.ndarray, seg_phi: np.ndarray):
    """ Transform array of phi into dict type with key nb.

        Parameters:
        --------
            seg_nb: np.ndarray
                Number of bands for each segment.
            seg_len: np.ndarray
                Length of each segment of time serials.
            seg_phi: np.ndarray
                Mean order parameter of each segment.

        Returns:
        --------
            Dict: dict
                Key = nb, value is also a dict with key sum_phi and count.
    """
    n_set = np.unique(seg_nb)
    dict_nb = {nb: {"sum_phi": 0, "count": 0} for nb in n_set if nb > 0}
    for i, nb in enumerate(seg_nb):
        if nb > 0:
            dict_nb[nb]["sum_phi"] += seg_len[i] * seg_phi[i]
            dict_nb[nb]["count"] += seg_len[i]
    return dict_nb


def get_dict_mean(dict0: dict, layer: int) -> dict:
    """ Get time (and sample) average of phi.

        Parameters:
        --------
            dict0: dict
                Sum of phi and count, with key seed or nb.
            level: int
                Layer of dict.
        Returns:
        --------
            dict1: dict
                Averaged phi and corresponding ratio.
    """
    dict1 = {}
    if layer == 2:
        tot = sum([dict0[nb]["count"] for nb in dict0])
        for nb in dict0:
            dict1[nb] = {nb: {"mean_phi": 0, "rate_phi": 0}}
            dict1[nb]["mean_phi"] = dict0[nb]["sum_phi"] / dict0[nb]["count"]
            dict1[nb]["rate_phi"] = dict0[nb]["count"] / tot
    elif layer == 3:
        tot = 0  # key = nb
        dict1 = {}  # key = nb
        for seed in dict0:
            for nb in dict0[seed]:
                if nb in tot:
                    dict1[nb]["mean_phi"] += dict0[seed][nb]["sum_phi"]
                    dict1[nb]["rate_phi"] += dict0[seed][nb]["count"]
                else:
                    dict1[nb]["mean_phi"] = dict0[seed][nb]["sum_phi"]
                    dict1[nb]["rate_phi"] = dict0[seed][nb]["count"]
                tot += dict0[seed][nb]["count"]
        for nb in tot:
            dict1[nb]["mean_phi"] /= tot
            dict1[nb]["rate_phi"] /= tot
    return dict1


def plot_phi(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        dict_sum = get_dict_sum(
            bf["seg_num"], bf["seg_idx1"] - bf["seg_idx0"], bf["seg_phi"])
        dict_mean = get_dict_mean(dict_sum, 2)
        for nb in dict_mean:
            rate = dict_mean[nb]["rate_phi"]
            if rate > 0.2:
                plt.scatter(Lx, dict_mean[nb]["mean_phi"], s=4, c=rate)
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


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    # file = "mb_350.0.740.200.214740.npz"
    # buff = np.load(file)
    # para = mb.get_para(file)
    # mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"],
    #                   buff["count_rhox"])
    plot_phi(350, 0)
