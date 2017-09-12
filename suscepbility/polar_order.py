"""
Estimate the correlation length beyond which the order parameter begin to decay
with a power law.
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from numpy.polynomial.polynomial import polyfit
from add_line import add_line
from suscept_peak import find_peak_by_polyfit
from fit import fit_exp


def read(file, dict_eps):
    eps = float(file.replace(".dat", ""))
    with open(file) as f:
        lines = f.readlines()
        L = np.zeros(len(lines))
        phi = np.zeros_like(L)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            L[i] = float(s[0])
            phi[i] = float(s[1])
        if L.size > 2 and eps >= 0.0535:
            dict_eps[eps] = {"L": L, "phi": phi}


def plot_phi_vs_L(ax=None, dict_eps=None, Lc=None, phi_c=None):
    """ PLot phi against L with varied epsilon in log-log scales."""
    if ax is None:
        ax = plt.gca()
        flag_show = True
    else:
        flag_show = False
    if dict_eps is None:
        dict_eps = {}
        files = glob.glob("0*.dat")
        for file in files:
            read(file, dict_eps)

    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(dict_eps)))
    for i, eps in enumerate(sorted(dict_eps.keys())):
        L = dict_eps[eps]["L"]
        phi = dict_eps[eps]["phi"]
        ax.plot(L, phi, "-o", label="%.4f" % eps, color=color[i])
    if Lc is not None and phi_c is not None:
        ax.plot(Lc, phi_c, "--ks", fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\phi$")
    ax.legend(title=r"$\epsilon=$", loc="lower left")
    add_line(ax, 0.25, 0.65, 0.55, -1, scale="log")
    if flag_show:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_L_vs_eps_c(alpha_list):
    dict_eps = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_eps)
    for alpha in alpha_list:
        c, res, eps_m, Lm, phi_m = find_peak(alpha, show=False, ret=True)
        plt.plot(eps_m, Lm, "-o")
    plt.yscale("log")
    plt.show()
    plt.close()


def find_peak(alpha,
              show=True,
              output=False,
              ax=None,
              dict_eps=None,
              ret=False):
    """ Find the peak of phi * L ** alpha against L in log-log scales."""
    if show:
        if ax is None:
            ax = plt.gca()
            flag_show = True
        else:
            flag_show = False
    if dict_eps is None:
        dict_eps = {}
        files = glob.glob("0*.dat")
        for file in files:
            read(file, dict_eps)
    xm = np.zeros(len(dict_eps.keys()))
    ym = np.zeros_like(xm)
    eps_m = np.zeros_like(xm)
    phi_m = np.zeros_like(xm)
    color = plt.cm.gist_rainbow(np.linspace(0, 1, xm.size))
    for i, key in enumerate(dict_eps):
        L = dict_eps[key]["L"]
        phi = dict_eps[key]["phi"]
        Y = phi * L**alpha
        xm[i], ym[i], x, y = find_peak_by_polyfit(
            L, Y, order=5, xscale="log", yscale="log", full=True)
        phi_m[i] = ym[i] / xm[i]**alpha
        eps_m[i] = key
        if show:
            ax.plot(L, Y, "o", label="%.4f" % key, color=color[i])
            ax.plot(x, y, color=color[i])

    c, stats = polyfit(np.log10(xm), np.log10(ym), 1, full=True)
    x = np.linspace(np.log10(xm[0]) + 0.05, np.log10(xm[-1]) - 0.05, 1000)
    y = c[0] + c[1] * x
    if output:
        with open("polar_order.dat", "w") as f:
            for i in range(eps_m.size):
                f.write("%.4f\t%.8f\n" % (eps_m[i], xm[i]))
    if show:
        ax.plot(xm, ym, "ks", fillstyle="none")
        ax.plot(10**x, 10**y, "k--", label=r"$L^{%.3f}$" % (c[1]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$L^{%g}\phi$" % alpha)
        # ax.legend()
        if flag_show:
            plt.tight_layout()
            plt.show()
            plt.close()
    if ret:
        return c[1], stats[0][0], eps_m, xm, phi_m


def plot_two_panel(alpha):
    dict_eps = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_eps)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    c0, res, eps_m, Lm, phi_m = find_peak(
        alpha, show=True, output=False, ax=ax2, dict_eps=dict_eps, ret=True)
    plot_phi_vs_L(ax1, dict_eps, Lm, phi_m)
    fig.tight_layout()
    plt.show()
    plt.close()
    plt.plot(eps_m, Lm, "-o")
    plt.yscale("log")
    plt.show()


def varied_alpha(nus):
    alpha = np.linspace(0.4, 0.6, 200)
    eps_c = np.zeros_like(alpha)
    lnA = np.zeros_like(alpha)
    b = np.zeros_like(alpha)
    std_eps_c = np.zeros_like(alpha)
    std_lnA = np.zeros_like(alpha)
    std_b = np.zeros_like(alpha)
    dict_eps = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_eps)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for nu in nus:
        for i in range(alpha.size):
            c, res, eps_m, Lm, phi_m = find_peak(
                alpha[i], show=False, ret=True, dict_eps=dict_eps)
            popt, perr = fit_exp(eps_m, Lm, beta=nu)
            eps_c[i], lnA[i], b[i] = popt
            std_eps_c[i], std_lnA[i], std_b[i] = perr
        ax1.plot(alpha, std_eps_c/eps_c)
        ax2.plot(alpha, std_lnA/lnA)
        ax3.plot(alpha, std_b/b)
    plt.show()


if __name__ == "__main__":
    os.chdir("data")
    varied_alpha([0.4, 0.45, 0.5, 0.55])
    # find_peak(0.5, output=True)
    # plot_two_panel(0.5)
    # plot_phi_vs_L()
    # plot_L_vs_eps_c([0.35, 0.4, 0.45, 0.5])
