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


def get_dict_phi_count(seg_nb: np.ndarray,
                       seg_len: np.ndarray,
                       seg_phi: np.ndarray):
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
            sum_phi_count: dict
                Key = nb, value = [\sum_nb seg_phi*seg_len, \sum_nb seg_len]
    """
    n_set = np.unique(seg_nb)
    sum_phi_count = {key: [0, 0] for key in n_set if key > 0}
    for i, nb in enumerate(seg_nb):
        if nb > 0:
            sum_phi_count[nb][0] += seg_len[i] * seg_phi[i]
            sum_phi_count[nb][1] += seg_len[i]
    return sum_phi_count


def plot_phi(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        sum_phi_count = get_dict_phi_count(
            bf["seg_num"], bf["seg_idx1"] - bf["seg_idx0"], bf["seg_phi"])
        tot = sum([sum_phi_count[key][1] for key in sum_phi_count])
        for k in sum_phi_count:
            mean = sum_phi_count[k][0] / sum_phi_count[k][1]
            p = sum_phi_count[k][1] / tot
            if p > 0.2:
                plt.scatter(Lx, mean, s=4, c=p)
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