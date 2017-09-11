"""
Estimate the susceptibility peak and its location from data.
Two type of averaging is used:
Type A:
    Find the susceptibiliyt peak and location for each sample, then average
    them.
Type B:
    Average susceptibilities over samples first, then find the peak of sample-
    averaged curve.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import interpolate
import os
import glob
import add_line


def read(file, dict_L):
    """ Read 0*.dat file.

    Parameters:
    --------
    file: str
        File to read, with pattern 0.*.dat
    dict_L: dict
        A dict with form {L:{eps: [phi, chi, err_chi, N]}}.
    """
    eps = float(file.replace(".dat", ""))
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            s = line.replace("\n", "").split("\t")
            L = int(s[0])
            phi = float(s[1])
            chi = float(s[4])
            err_chi = float(s[5])
            N = int(s[3])
            if L in dict_L:
                dict_L[L][eps] = [phi, chi, err_chi, N]
            else:
                dict_L[L] = {eps: [phi, chi, err_chi, N]}


def read_npz(L):
    """ Read "%d.npz" % L, and print sample-averaged values.

    Parameters:
    --------
    L : int
        System size.
    """
    data = np.load("%d.npz" % L)
    chi = data["chi"]
    phi = data["phi"]
    eps = data["eps"] / 10000
    sample_size = chi.shape[0]
    chi_mean = np.mean(chi, axis=0)
    chi_std = np.std(chi, axis=0)
    phi_mean = np.mean(phi, axis=0)
    phi_std = np.std(phi, axis=0)
    with open("tmp.dat", "w") as f:
        for i in range(chi_mean.size):
            f.write("%f\n" % (eps[i]))
            f.write("%d\t%f\t%f\t%d\t%f\t%f\n" %
                    (L, phi_mean[i], phi_std[i], sample_size, chi_mean[i],
                     chi_std[i]))


def find_peak(eps, xi, order=5, full=False):
    c = polyfit(eps, np.log10(xi), order)
    x = np.linspace(eps.min(), eps.max(), 10000)
    y = c[0] * np.ones(x.size)
    for i in range(1, order + 1):
        y += c[i] * x**i
    idx = np.argmax(y)
    if full:
        return x[idx], 10**(y[idx]), x, 10**y
    else:
        return x[idx], 10**(y[idx])


def find_max_by_spline(xdata, ydata):
    f = interpolate.interp1d(xdata, ydata, kind="cubic")
    x = np.linspace(xdata.min(), xdata.max(), 2000)
    y = f(x)
    idxm = y.argmax()
    xm = x[idxm]
    ym = y[idxm]
    return xm, ym


def diff_find_peak(L, *idx):
    """ Show the effects of different fitting.

    For a given L, we fit the curve of chi vs. eps, and find where chi gets
    its maximum value.

    Parameters:
    --------
    L : int
        System size.
    *idx : argment list
        List of sample to show.
    """
    data = np.load("%d.npz" % L)
    chi0 = data["chi"]
    eps0 = data["eps"] / 10000

    for j, i in enumerate(idx):
        line, = plt.plot(eps0, chi0[i], "o", label="Sample %d" % j)
        c = line.get_color()
        eps_p, chi_p, eps, chi = find_peak(eps0, chi0[i], order=3, full=True)
        plt.plot(eps, chi, linestyle="dashed", color=c)
        eps_p, chi_p, eps, chi = find_peak(eps0, chi0[i], order=5, full=True)
        plt.plot(eps, chi, linestyle="solid", color=c)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\chi$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def varied_sample_size(L, M):
    """ Show how the difference between two methods changes with increasing
        sample size.

    Parameterss:
    --------
    L : int
        System size
    M : int
        Max sample size.
    """
    data = np.load("%d.npz" % L)
    chi = data["chi"]
    eps = data["eps"] / 10000

    N = (np.arange(M // 50) + 1) * 50
    eps_p1 = np.zeros(N.size)
    eps_p2 = np.zeros(N.size)
    chi_p1 = np.zeros(N.size)
    chi_p2 = np.zeros(N.size)

    for i, n in enumerate(N):
        eps_p = np.zeros(n)
        chi_p = np.zeros(n)
        for j in range(n):
            eps_p[j], chi_p[j] = find_peak(eps, chi[j], order=5)
        eps_p1[i] = np.mean(eps_p)
        chi_p1[i] = np.mean(chi_p)

        chi_mean = np.mean(chi[:n], axis=0)
        eps_p2[i], chi_p2[i] = find_peak(eps, chi_mean, order=5)

    plt.subplot(221)
    plt.plot(N, chi_p1, "-o", label=r"$\langle \chi_m\rangle^A_s$")
    plt.plot(N, chi_p2, "-s", label=r"$\langle \chi_m\rangle^B_s$")
    plt.xlabel("Sample size")
    plt.ylabel("Susceptibility peak")
    plt.legend()

    plt.subplot(222)
    plt.plot(N, eps_p1, "-o", label=r"$\langle \epsilon_m\rangle^A_s$")
    plt.plot(N, eps_p2, "-s", label=r"$\langle \epsilon_m\rangle^B_s$")
    plt.xlabel("Sample size")
    plt.ylabel("Peak location")
    plt.legend()

    plt.subplot(223)
    plt.plot(N, (chi_p1 - chi_p2) / chi_p2, "k-o")
    plt.xlabel("Sample size")
    ylabel = r"$|\langle \chi_m\rangle^A_s - \langle\chi_m\rangle^B_s|"\
        + r"/ \langle \chi_m\rangle ^B_s$"
    plt.ylabel(ylabel)

    plt.subplot(224)
    plt.plot(N, -(eps_p1 - eps_p2) / eps_p2, "k-o")
    plt.xlabel("Sample size")
    ylabel = r"$|\langle\epsilon_m\rangle^A_s - \langle\epsilon_m\rangle^B_s|"\
        + r"/ \langle\epsilon_m\rangle^B_s$"
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()
    plt.close()


def distrubition(L, N):
    """ Show the distrubition of susceptibility peaks and locations estimated
        from each sample.
    """
    data = np.load("%d.npz" % L)
    chi = data["chi"]
    eps = data["eps"] / 10000
    chi_m, eps_m, chi_m2, eps_m2 = np.zeros((4, N))
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    for i in range(N):
        eps_m[i], chi_m[i] = find_peak(eps, chi[i], order=5)
        eps_m2[i], chi_m2[i] = find_max_by_spline(eps, chi[i])
    # chi_m2 = np.exp(chi_m2)
    ax[0].plot(eps_m, chi_m, "o", markersize=3)
    ax[0].set_xlabel(r"$\epsilon_m$")
    ax[0].set_ylabel(r"$\chi_m$")
    # ax[0].set_ylim(ymax=250)
    # plt.yscale("log")
    hist_eps, bin_eps = np.histogram(eps_m, bins=10)
    ax[1].plot(0.5 * (bin_eps[:-1] + bin_eps[1:]),
               hist_eps / N / (bin_eps[1] - bin_eps[0]), "-o")
    ax[1].set_xlabel(r"$\epsilon_m$")
    ax[1].set_ylabel(r"PDF$(\epsilon_m)$")

    hist_xi, bin_xi = np.histogram(chi_m, bins=10)
    ax[2].plot(0.5 * (bin_xi[:-1] + bin_xi[1:]),
               hist_xi / N / (bin_xi[1] - bin_xi[0]), "-o")
    ax[2].set_xlabel(r"$\chi_m$")
    ax[2].set_ylabel(r"PDF$(\chi_m)$")
    plt.tight_layout()
    plt.show()
    plt.close()

    print("A: chi_m=%f\teps_m=%f" % (np.mean(chi_m), np.mean(eps_m)))
    chi_mean = np.mean(chi, axis=0)
    eps_peak, chi_peak = find_peak(eps, chi_mean, order=5)
    eps_peak2, chi_peak2 = find_max_by_spline(eps, chi_mean)
    print("B: chi_m=%f\teps_m=%f" % (chi_peak, eps_peak))
    print("C: chi_m=%f\teps_m=%f" % (np.mean(chi_m2), np.mean(eps_m2)))

    plt.plot(eps, chi_mean, "-o")
    plt.plot(eps_peak, chi_peak, "s", eps_peak2, chi_peak2, "<")
    plt.plot(np.mean(eps_m), np.mean(chi_m), "p")
    plt.plot(np.mean(eps_m2), np.mean(chi_m2), ">")
    plt.yscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Sample-averaged $\chi$")
    plt.tight_layout()
    plt.show()
    plt.close()


def average_type_B(save=False):
    """ Calculate the susceptibility peak by "B" type averaging.
    """
    dict_L = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_L)
    # Plot susceptibility peak vs its location.
    L_arr = np.array([46, 64, 90, 128, 180, 256, 362, 512, 724])
    eps_p, chi_p = np.zeros((2, len(L_arr)))
    for L in sorted(dict_L.keys()):
        if L in L_arr:
            chi_list = []
            eps_list = []
            for eps in dict_L[L]:
                if eps >= 0.045:
                    chi_list.append(dict_L[L][eps][1])
                    eps_list.append(eps)
            chi = np.array(chi_list)
            # err_xi = np.array(err_xi_list)
            eps = np.array(eps_list)
            line, = plt.plot(eps, chi, "o", label=r"$%d$" % L)
            i = np.argwhere(L_arr == L)
            eps_p[i], chi_p[i], x, y = find_peak(eps, chi, 5, True)
            plt.plot(x, y, color=line.get_color())

    plt.plot(eps_p, chi_p, "ks--", fillstyle="none")
    plt.yscale("log")
    plt.legend(loc="upper right", title=r"$L=$")
    plt.xlim(xmax=0.09)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Sample-averaged susceptibility")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Save data
    if save:
        with open("suscept_peak.dat", "w") as f:
            for i, L in enumerate(L_arr):
                f.write("%d\t%.8f\t%.8f\n" % (L, eps_p[i], chi_p[i]))

    # Plot susceptibility peak and location vs. system size, respectively.
    plot_peak_location_vs_L(L_arr, eps_p, chi_p)


def plot_peak_location_vs_L(L=None, eps_p=None, chi_p=None):
    """ Plot susceptibility peak and its loaction against system size,
        respectively.

    Parameters:
    --------
    L : array_like, optional
        Array of system sizes.
    eps_p : array_like, optional
        Locations of susceptibility peak.
    chi_p : array_like, optional
        Values of susceptibility peak.
    """
    if L is None or eps_p is None or chi_p is None:
        with open("suscept_peak.dat") as f:
            lines = f.readlines()
            L, eps_p, chi_p = np.zeros((3, len(lines)))
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                L[i] = float(s[0])
                eps_p[i] = float(s[1])
                chi_p[i] = float(s[2])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].loglog(L, chi_p, "o")
    c = polyfit(np.log10(L), np.log10(chi_p), deg=1)
    x0, x1 = ax[0].get_xlim()
    x = np.linspace(x0, x1, 1000)
    y = 10**c[0] * x**c[1]
    ax[0].loglog(
        x, y, "--", label=r"$\langle \chi_m\rangle^B_s\sim L^{%.4f}$" % c[1])
    ax[0].set_xlim(x0, x1)
    ax[0].legend(loc="lower right", title="liner fit")
    add_line.add_line(ax[0], 0, 0.2, 1, 1.75, label=r"$L^{1.75}$", scale="log")
    ax[0].set_xlabel(r"$L$")
    ax[0].set_ylabel(r"${\rm susceptibility\ peak\ }\langle\chi_m\rangle^B_s$")

    ax[1].loglog(L, eps_p, "-o")
    ax[1].set_xlabel(r"$L$")
    ax[1].set_ylabel(r"${\rm peak\ location\ }\langle \epsilon_m\rangle^B_s$")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data")
    # distrubition(90, 500)
    # varied_sample_size(90, 500)
    average_type_B(True)
    # diff_find_peak(90, 4, 10)
    # plot_peak_location_vs_L()
    # read_npz(724)
