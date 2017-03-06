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


def plot_phi(eta, eps):
    files = glob.glob("mb_%d.%d.*.npz" % (eta, eps))
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        Lx = para[2]
        phi = {key: 0 for key in bf["num_set"] if key > 0}
        count = {key: 0 for key in phi}
        for i in range(bf["seg_num"].size):
            k = bf["seg_num"][i]
            if k > 0:
                dt = bf["seg_idx1"][i] - bf["seg_idx0"][i]
                phi[k] += bf["seg_phi"][i] * dt
                count[k] += dt
        tot = sum([count[k] for k in count])
        for k in phi:
            mean = phi[k] / count[k]
            p = count[k] / tot
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
                plt.plot(1/Lx, nb/Lx, "o")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    # file = "mb_350.0.740.200.214740.npz"
    # buff = np.load(file)
    # para = mb.get_para(file)
    # mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"],
    #                   buff["count_rhox"])
    plot_nb(350, 20)
