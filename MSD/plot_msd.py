import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
sys.path.append("..")
try:
    from corr2d.add_line import add_line
except ImportError:
    print("error when import add_line")


def readMSD(file):
    with open(file) as f:
        lines = f.readlines()
        n = len(lines)
        t = np.zeros(n)
        msd = np.zeros(n)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            t[i] = int(s[0])
            msd[i] = float(s[1])
    return t, msd


def plotMSD(eta, L=None, eps=None, disorder="RT", seed=123,
            ax=None, excluding_eps=[]):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))
        flag_show = True
    else:
        flag_show = False
    if L is not None:
        files = glob.glob("msd_%s_%d_%g_*_%d.dat" % (disorder, L, eta, seed))
        label_pat = r"$%.2f$"
        legend_title = r"$\epsilon=$"
    elif eps is not None:
        files = glob.glob("msd_%s_*_%g_%g_%d.dat" % (disorder, eta, eps, seed))
        label_pat = r"$%d$"
        legend_title = r"$L=$"
    msd_dict = {}
    for file in files:
        t, msd = readMSD(file)
        if L is not None:
            key = float(file.split("_")[4])
            if key not in excluding_eps:
                msd_dict[key] = [t, msd]
        elif eps is not None:
            key = int(file.split("_")[2])
            msd_dict[key] = [t, msd]
    for key in sorted(msd_dict.keys()):
        t, msd = msd_dict[key]
        ax.plot(t, msd, label=label_pat % key)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$", fontsize="x-large")
    ax.set_ylabel("MSD", fontsize="x-large")
    if L is not None:
        title = r"%s: $\eta=%g, L=%d$" % (disorder, eta, L)
    elif eps is not None:
        title = r"%s: $\eta=%g, \epsilon=%.2f$" % (disorder, eta, eps)
    ax.set_title(title, fontsize="x-large")
    ax.legend(title=legend_title)
    if flag_show:
        add_line(ax, 0.1, 0, 0.5, slope=1, label=r"$\propto t$", yl=0.05, scale="log")
        add_line(ax, 0.5, 0.5, 1, slope=2, label=r"$\propto t^2$", scale="log")
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_6_MSD(L, seed=123):
    if L == 32:
        disorder_list = ["RT", "RF"]
        eta_list = [0, 0.1, 0.5]
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
        for i, disorder in enumerate(disorder_list):
            for j, eta in enumerate(eta_list):
                plotMSD(eta, L=L, disorder=disorder, ax=axes[i][j])
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_4_MSD(L, seed=123):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    if L == 32:
        disorder_list = ["RT", "RF"]
        eta_list = [0., 0.1]
        excluding_eps = {}
        excluding_eps["RT"] = {
            0.: [0.03, 0.09, 0.15, 0.18, 0.24, 0.4],
            0.1: [0.09, 0.15, 0.18, 0.21, 0.24]
        }
        excluding_eps["RF"] = {
            0.: [0.03, 0.09, 0.12, 0.15, 0.21, 0.24, 0.27, 0.35, 0.45, 0.55, 0.65],
            0.1: [0.06, 0.09, 0.12, 0.15, 0.21, 0.24, 0.27, 0.35, 0.45, 0.55, 0.65, 1]
        }
        for i, disorder in enumerate(disorder_list):
            for j, eta in enumerate(eta_list):
                plotMSD(eta, L=L, disorder=disorder, ax= axes[i][j],
                        excluding_eps=excluding_eps[disorder][eta])
    add_line(axes[0][0], 0.05, 0, 0.5, slope=1, label=r"$\sim t$", yl=0.08, scale="log")
    add_line(axes[0][0], 0.5, 0.35, 0.95, slope=2, label=r"$\sim t^2$", xl=0.7, yl=0.65, scale="log")
    add_line(axes[0][1], 0.05, 0, 0.6, slope=1, label=r"$\sim t$", yl=0.04, scale="log")
    add_line(axes[0][1], 0, 0.2, 0.95, slope=2, label=r"$\sim t^2$", xl=0.5, yl=0.8, scale="log")
    add_line(axes[1][0], 0.05, 0, 0.6, slope=1, label=r"$\sim t$", yl=0.04, scale="log")
    add_line(axes[1][0], 0, 0.2, 0.95, slope=2, label=r"$\sim t^2$", xl=0.45, yl=0.8, scale="log")
    add_line(axes[1][1], 0.05, 0, 0.6, slope=1, label=r"$\sim t$", yl=0.04, scale="log")
    add_line(axes[1][1], 0, 0.2, 0.95, slope=2, label=r"$\sim t^2$", xl=0.5, yl=0.8, scale="log")
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    os.chdir(r"D:\data\random_field\normalize_new\MSD_new")
    plotMSD(0, eps=0.15, L=None, disorder="RT")
    # plot_6_MSD(32)
    # plot_4_MSD(32)
