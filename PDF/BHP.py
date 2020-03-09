"""
    Cal the PDF of time-averaged order parameters among different samples.

    Cal the PDF of instant order parameters for a single sample.

    Compare the obtain PDF with the BHP and Gaussian distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def BHP_approx(y, b=0.938, s=0.374, K=2.14):
    a = np.pi / 2
    x = b * (y - s)
    return K * (np.exp(x - np.exp(x)))**a


def read_samples(eta, eps, L, d=2, disorder_type="RT"):
    if d == 2:
        if disorder_type == "RT":
            os.chdir(
                r"E:\data\random_torque\susceptibility\time_average\eta=%.2f" %
                eta)
            file_pat = r"%.2f_%.4f_%d.xlsx"
        else:
            os.chdir(r"E:\data\random_field\normalize\scaling\time_average")
            file_pat = r"%.3f_%.4f_%d.xlsx"
        df = pd.read_excel(file_pat % (eta, eps, L), sheet_name="Sheet1")
        phi_arr = df["mean"]
    else:
        if disorder_type == "RT":
            os.chdir(r"E:\data\vm3d")
            file_name = r"phi3_%d_%.2f_%.3f.dat" % (L, eta, eps)
            with open(file_name, "r") as f:
                lines = f.readlines()
                phi_arr = np.array([float(i.replace("\n", "")) for i in lines])
    return phi_arr


def read_time_serials(eta,
                      eps,
                      L,
                      seed,
                      d=2,
                      disorder_type="RT",
                      ncut=5000,
                      fin=None):
    print("ncut =", ncut)
    if fin is None:
        if d == 2:
            if disorder_type == "RT":
                if eta == 0.1:
                    dest = r"E:\data\random_torque\Phi_vs_L\eta=0.10\serials"
                elif eta == 0.18:
                    dest = r"E:\data\random_torque\Phi_vs_L\eta=0.18\%.3f" % eps
                print(dest)
                os.chdir(dest)
                filename = "p%d.%d.%d.%d.dat" % (L, int(eta * 1000),
                                                 int(eps * 10000), seed)
            elif disorder_type == "RF":
                os.chdir(r"E:\data\random_field\normalize_new\scaling\serials")
                filename = "phi_rf_%d_%.2f_%.2f_%d.dat" % (L, eta, eps, seed)
        elif d == 3:
            if disorder_type == "RT":
                os.chdir(r"E:\data\vm3d\order_para")
                filename = "phi_%d_%.2f_%.3f_1.0_%d.dat" % (L, eta, eps, seed)
    else:
        filename = fin
    print(filename)
    with open(filename, "r") as f:
        lines = f.readlines()[ncut:]
        phi_arr = np.array([float(i.split("\t")[0]) for i in lines])
    return phi_arr


def plot_PDF(eta,
             eps,
             L,
             d,
             disorder_type,
             ax=None,
             bins=13,
             mk='o',
             seed=None,
             filename=None,
             ncut=5000):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        flag_show = True
    else:
        flag_show = False
    if filename is None:
        if seed is None:
            phi_arr = read_samples(eta, eps, L, d, disorder_type)
        else:
            phi_arr = read_time_serials(eta, eps, L, seed, d, disorder_type)
    else:
        phi_arr = read_time_serials(
            eta, eps, L, seed, d, disorder_type, ncut=ncut, fin=filename)
    phi_mean = np.mean(phi_arr)
    phi_std = np.std(phi_arr)
    hist, bin_edges = np.histogram(phi_arr, bins=bins, density=True)
    x = ((bin_edges[:-1] + bin_edges[1:]) * 0.5 - phi_mean) / phi_std
    y = hist * phi_std
    ax.plot(x, y, mk, label=r"$L=%d, \sigma=%.4f$" % (L, phi_std), alpha=0.7)
    if flag_show:
        y = np.linspace(-6, 2.5, 1000)
        f = BHP_approx(y)
        ax.plot(y, f)
        y = np.linspace(-4, 3.5, 1000)
        f = 1 / np.sqrt(2 * np.pi) * np.exp(-y**2 / 2)
        ax.plot(y, f, "--")
        ax.set_yscale("log")
        plt.show()
        plt.close()


def plot_PDF_mult_samples():
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 4), sharey=True)
    y = np.linspace(-6, 2.5, 1000)
    f = BHP_approx(y)
    ax1.plot(y, f)
    ax2.plot(y, f)
    ax3.plot(y, f)

    y = np.linspace(-4, 2.6, 1000)
    f = 1 / np.sqrt(2 * np.pi) * np.exp(-y**2 / 2)
    ax1.plot(y, f, "--")
    ax2.plot(y, f, "--")
    ax3.plot(y, f, "--")

    plot_PDF(0.18, 0.02, 32, 2, "RT", ax1, 13, "o")
    plot_PDF(0.18, 0.02, 90, 2, "RT", ax1, 13, "D")
    plot_PDF(0.18, 0.02, 256, 2, "RT", ax1, 13, "v")
    plot_PDF(0.18, 0.02, 512, 2, "RT", ax1, 13, "s")
    plot_PDF(0.18, 0.08, 362, 2, "RF", ax2, 13, "o")
    plot_PDF(0.18, 0.08, 512, 2, "RF", ax2, 13, "s")
    plot_PDF(0.2, 0.12, 16, 3, "RT", ax3, 13, "o")
    plot_PDF(0.2, 0.12, 22, 3, "RT", ax3, 12, "s")
    ax1.set_title(r"$\eta=0.18, \epsilon=0.035, d=2$, RT", fontsize="large")
    ax2.set_title(r"$\eta=0.18, \epsilon=0.08, d=2$, RF", fontsize="large")
    ax3.set_title(r"$\eta=0.2, \epsilon=0.12, d=3$, RT", fontsize="large")

    ax1.legend(fontsize="medium")
    ax2.legend(fontsize="medium")
    ax3.legend(fontsize="medium")

    xlabel = r"$[\langle\phi \rangle - \overline{\langle \phi\rangle}]/\sigma$"

    ax1.set_xlabel(xlabel, fontsize="large")
    ax2.set_xlabel(xlabel, fontsize="large")
    ax3.set_xlabel(xlabel, fontsize="large")
    ax1.set_ylabel(r"$f\times \sigma$", fontsize="large")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_PDF_single_samples():
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(10, 4), sharey=True)
    y = np.linspace(-5.5, 2.5, 1000)
    f = BHP_approx(y)
    ax1.plot(y, f)
    ax2.plot(y, f)
    ax3.plot(y, f)

    y = np.linspace(-4, 3, 1000)
    f = 1 / np.sqrt(2 * np.pi) * np.exp(-y**2 / 2)
    ax1.plot(y, f, "--")
    ax2.plot(y, f, "--")
    ax3.plot(y, f, "--")

    plot_PDF(0.18, 0.02, 180, 2, "RT", seed=201292, bins=15, ax=ax1, mk="o")
    plot_PDF(0.18, 0.02, 362, 2, "RT", seed=3836952, bins=15, ax=ax1, mk="s")
    plot_PDF(0.18, 0.02, 724, 2, "RT", seed=25440689, bins=15, ax=ax1, mk=">")

    plot_PDF(0.18, 0.07, 128, 2, "RF", seed=14420000, bins=15, ax=ax2, mk="o")
    plot_PDF(0.18, 0.07, 256, 2, "RF", seed=15700050, bins=15, ax=ax2, mk="s")
    plot_PDF(0.18, 0.07, 512, 2, "RF", seed=18260000, bins=15, ax=ax2, mk=">")

    plot_PDF(0.2, 0.12, 120, 3, "RT", seed=440, bins=15, ax=ax3, mk="o")
    plot_PDF(0.2, 0.12, 240, 3, "RT", seed=21, bins=15, ax=ax3, mk="s")

    ax1.set_title(r"$\eta=0.18, \epsilon=0.02, d=2$, RT", fontsize="large")
    ax2.set_title(r"$\eta=0.18, \epsilon=0.07, d=2$, RF", fontsize="large")
    ax3.set_title(r"$\eta=0.2, \epsilon=0.12, d=3$, RT", fontsize="large")

    ax1.legend(fontsize="medium")
    ax2.legend(fontsize="medium")
    ax3.legend(fontsize="medium")
    xlabel = r"$\left (\phi - \langle \phi\rangle\right ) / \sigma$"
    ax1.set_xlabel(xlabel, fontsize="large")
    ax2.set_xlabel(xlabel, fontsize="large")
    ax3.set_xlabel(xlabel, fontsize="large")
    ax1.set_ylabel(r"$f\times \sigma$", fontsize="large")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_PDF_mult_samples()
    # plot_PDF_single_samples()
    # y = np.linspace(-10, 10, 1000)
    # f = BHP_approx(y)
    # plt.plot(y, f)
    # y = np.linspace(-10, 10, 1000)
    # f = 1 / np.sqrt(2 * np.pi) * np.exp(-y**2 / 2)
    # plt.plot(y, f, "--")
    # plt.yscale("log")
    # plt.show()
    # plt.close()
    ''' distribution for the case of bands '''
    # f1 = 'E:/data/random_torque/bands/Lx/snapshot/eps0/p350.0.400.200.214400.dat'
    # f2 = 'E:/data/random_torque/bands/Lx/snapshot/eps20/p350.30.440.200.7440.dat'

    # f1 = 'E:/data/random_torque/Phi_vs_L/eta=0.18/0.000/p2048.180.0.25984.dat'
    # f2 = 'E:/data/random_torque/Phi_vs_L/eta=0.18/0.000/p2048.180.0.20944.dat'

    # f1 = r"E:\data\random_torque\large_system\phi_2400_0.18_0.000_1.0_1.dat"
    # f2 = r"E:\data\random_torque\large_system\phi_4800_0.18_0.020_1.0_1.dat"

    # f1 = r"E:\data\random_torque\statistic\2048\serials\s_0.180_0.000_2048_1257.dat"
    # f2 = r"E:\data\random_torque\statistic\2048\serials\s_0.180_0.010_2048_229.dat"

    # f1 = r"E:\data\random_potentail\l=1\order_para\phi_0.060_0.100_1024_123.dat"
    # f2 = r"E:\data\random_potentail\l=1\order_para\phi_0.140_0.100_1024_123.dat"
    # f3 = r"E:\data\random_potentail\l=1\order_para\phi_0.320_0.100_1024_123.dat"


    f1 = r"G:\data\vm3d\vm3d_eps=0.06_eta=0.2\data\phi_120_0.20_0.060_1.0_930006.dat"
    plt.figure(constrained_layout=True)
    ax = plt.gca()

    ncut = 5000
    plot_PDF(0.2, 0.06, 120, 3, "RT", filename=f1, ax=ax, ncut=ncut)
    # plot_PDF(
    #     0.35,
    #     0,
    #     350,
    #     2,
    #     "RT",
    #     seed=214400,
    #     filename=f2,
    #     ax=ax,
    #     mk="s",
    #     ncut=ncut)
    # plot_PDF(0.35, 0, 350, 2, "RT", seed=214400, filename=f3, ax=ax, mk=">", ncut=ncut)

    y = np.linspace(-6, 2.5, 1000)
    f = BHP_approx(y)
    ax.plot(y, f)
    y = np.linspace(-4, 3.5, 1000)
    f = 1 / np.sqrt(2 * np.pi) * np.exp(-y**2 / 2)
    ax.plot(y, f, "--")
    plt.yscale("log")
    xlabel = r"$\left (\phi - \langle \phi\rangle\right ) / \sigma$"
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel(r"$f\times \sigma$", fontsize="large")
    plt.show()
    plt.close()
