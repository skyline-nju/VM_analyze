import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def BHP_approx(y, b=0.938, s=0.374, K=2.14):
    a = np.pi / 2
    x = b * (y - s)
    return K * (np.exp(x - np.exp(x))) ** a


def read_data(eta, eps, L, d=2, disorder_type="RT"):
    if d == 2:
        if disorder_type == "RT":
            os.chdir(r"D:\data\random_torque\susceptibility\time_average\eta=%.2f" % eta)
            file_pat = r"%.2f_%.4f_%d.xlsx"
        else:
            os.chdir(r"D:\data\random_field\normalize\scaling\time_average")
            file_pat = r"%.3f_%.4f_%d.xlsx"
        df = pd.read_excel(file_pat % (eta, eps, L), sheet_name="Sheet1")
        phi_arr = df["mean"]
    else:
        if disorder_type == "RT":
            os.chdir(r"D:\data\vm3d")
            file_name = r"phi3_%d_%.2f_%.3f.dat" % (L, eta, eps)
            with open(file_name, "r") as f:
                lines = f.readlines()
                phi_arr = np.array([float(i.replace("\n", "")) for i in lines])
    return phi_arr


def plot_PDF(eta, eps, L, d, disorder_type, ax, bins=13):
    phi_arr = read_data(eta, eps, L, d, disorder_type)
    phi_mean = np.mean(phi_arr)
    phi_std = np.std(phi_arr)
    hist, bin_edges = np.histogram(phi_arr, bins=bins, density=True)
    x = ((bin_edges[:-1] + bin_edges[1:]) * 0.5 - phi_mean) / phi_std
    y = hist * phi_std
    ax.plot(x, y, "o", label=r"$L=%d, \sigma=%.4f$" % (L, phi_std))


def plot_all_PDF():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharey=True)
    y = np.linspace(-6, 2, 1000)
    f = BHP_approx(y)
    ax1.plot(y, f)
    ax2.plot(y, f)
    ax3.plot(y, f)

    y = np.linspace(-4, 2.6, 1000)
    f = 1 / np.sqrt(2 * np.pi) * np.exp(- y ** 2 / 2)
    ax3.plot(y, f, "--")

    plot_PDF(0.18, 0.035, 128, 2, "RT", ax1, 13)
    plot_PDF(0.18, 0.035, 512, 2, "RT", ax1, 13)
    plot_PDF(0.18, 0.08, 362, 2, "RF", ax2, 13)
    plot_PDF(0.18, 0.08, 512, 2, "RF", ax2, 13)
    plot_PDF(0.2, 0.12, 16, 3, "RT", ax3, 13)
    plot_PDF(0.2, 0.12, 22, 3, "RT", ax3, 13)
    ax1.set_title(r"$\eta=0.18, \epsilon=0.035, d=2$, RT", fontsize="large")
    ax2.set_title(r"$\eta=0.18, \epsilon=0.08, d=2$, RF", fontsize="large")
    ax3.set_title(r"$\eta=0.2, \epsilon=0.12, d=3$, RT", fontsize="large")

    ax1.legend(fontsize="large")
    ax2.legend(fontsize="large")
    ax3.legend(fontsize="large")

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


if __name__ == "__main__":
    y = np.linspace(-6, 2, 1000)
    f = BHP_approx(y)
    # plt.plot(y, f)
    # # phi = read_data(0.18, 0.08, 512, type="RF")
    # # phi = read_data(0.18, 0.035, 362)
    # phi = read_data(0.2, 0.12, 22, d=3 )
    # plot_PDF(phi, plt.gca())
    # plt.yscale("log")
    # plt.show()
    # plt.close()
    plot_all_PDF()
