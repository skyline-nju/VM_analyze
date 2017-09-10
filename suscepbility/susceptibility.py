import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import interpolate
import os
import glob
import add_line


def read(file, dict_L):
    eps = float(file.replace(".dat", ""))
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            s = line.replace("\n", "").split("\t")
            L = int(s[0])
            phi = float(s[1])
            xi = float(s[4])
            err_xi = float(s[5])
            N = int(s[3])
            if L in dict_L:
                dict_L[L][eps] = [phi, xi, err_xi, N]
            else:
                dict_L[L] = {eps: [phi, xi, err_xi, N]}


def fit(eps, xi, ax=None, label=None):
    if ax is None:
        ax = plt.subplot()
        flag_show = True
    else:
        flag_show = False
    line, = ax.plot(eps, xi, "o")
    if label is not None:
        line.set_label(label)
    x = np.linspace(eps.min(), eps.max(), 1000)
    c = polyfit(eps, np.log10(xi), 3)
    y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
    y = 10**y
    ax.plot(x, y, color=line.get_color(), linestyle='dashed')
    c = polyfit(eps, np.log10(xi), 5)
    y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4 + c[5] * x**5
    y = 10**y
    ax.plot(x, y, color=line.get_color(), linestyle='solid')
    ax.set_yscale("log")
    if flag_show:
        plt.yscale("log")
        plt.show()


def find_peak(eps, xi, order=5):
    c = polyfit(eps, np.log10(xi), order)
    x = np.linspace(eps.min(), eps.max(), 10000)
    y = c[0] * np.ones(x.size)
    for i in range(1, order + 1):
        y += c[i] * x**i
    idx = np.argmax(y)
    return x[idx], 10**(y[idx])


def varied_sample_size(L, M):
    data = np.load("%d.npz" % L)
    # phi = data["phi"]
    xi = data["xi"]
    eps = data["eps"] / 10000

    N = (np.arange(M // 50) + 1) * 50
    eps_m1 = np.zeros(N.size)
    eps_m2 = np.zeros(N.size)
    chi_m1 = np.zeros(N.size)
    chi_m2 = np.zeros(N.size)

    for i, n in enumerate(N):
        epsilon = np.zeros(n)
        chi = np.zeros(n)
        for j in range(n):
            epsilon[j], chi[j] = find_peak(eps, xi[j], order=5)
        eps_m1[i] = np.mean(epsilon)
        chi_m1[i] = np.mean(chi)

        chi_mean = np.mean(xi[:n], axis=0)
        eps_m2[i], chi_m2[i] = find_peak(eps, chi_mean, order=5)

    plt.subplot(221)
    plt.plot(N, chi_m1, "-o", label=r"$\langle \chi_m\rangle^A_s$")
    plt.plot(N, chi_m2, "-s", label=r"$\langle \chi_m\rangle^B_s$")
    plt.xlabel("Sample size")
    plt.ylabel("Susceptibility peak")
    plt.legend()

    plt.subplot(222)
    plt.plot(N, eps_m1, "-o", label=r"$\langle \epsilon_m\rangle^A_s$")
    plt.plot(N, eps_m2, "-s", label=r"$\langle \epsilon_m\rangle^B_s$")
    plt.xlabel("Sample size")
    plt.ylabel("Peak location")
    plt.legend()

    plt.subplot(223)
    plt.plot(N, (chi_m1 - chi_m2) / chi_m2, "k-o")
    plt.xlabel("Sample size")
    plt.ylabel(
        r"$|\langle \chi_m\rangle^A_s - \langle\chi_m\rangle^B_s| / \langle \chi_m\rangle ^B_s$"
    )

    plt.subplot(224)
    plt.plot(N, -(eps_m1 - eps_m2) / eps_m2, "k-o")
    plt.xlabel("Sample size")
    plt.ylabel(
        r"$|\langle\epsilon_m\rangle^A_s - \langle\epsilon_m\rangle^B_s| / \langle\epsilon_m\rangle^B_s$"
    )

    plt.tight_layout()
    plt.show()
    plt.close()


def distrubition(L, N):
    data = np.load("%d.npz" % L)
    chi = data["xi"]
    eps = data["eps"] / 10000

    ax = plt.subplot(111)
    fit(eps, chi[47], ax, "sample A")
    fit(eps, chi[48], ax, "sample B")
    fit(eps, chi[49], ax, "sample C")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\chi$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    chi_m = np.zeros(N)
    eps_m = np.zeros(N)
    chi_m2 = np.zeros(N)
    eps_m2 = np.zeros(N)
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
    dict_L = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_L)
    # Plot susceptibility peak against corresponding location.
    L_list = [46, 64, 90, 128, 180, 256, 362, 512, 724]
    eps_c = []
    peak = []
    eps_c2 = []
    peak2 = []
    for L in sorted(dict_L.keys()):
        if L in L_list:
            xi_list = []
            err_xi_list = []
            eps_list = []
            for eps in dict_L[L]:
                if eps >= 0.045:
                    xi_list.append(dict_L[L][eps][1])
                    err_xi_list.append(dict_L[L][eps][2])
                    eps_list.append(eps)
            xi = np.array(xi_list)
            # err_xi = np.array(err_xi_list)
            eps = np.array(eps_list)
            line, = plt.plot(eps, xi, "o", label=r"$%d$" % L)
            c = np.polynomial.polynomial.polyfit(eps, np.log10(xi), 5)
            x = np.linspace(eps.min(), eps.max(), 1000)
            y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4 + c[5] * x**5
            y = 10**y
            plt.plot(x, y, color=line.get_color())
            idx = np.argmax(y)
            peak.append(y[idx])
            eps_c.append(x[idx])
            a, b = find_max_by_spline(eps, xi)
            eps_c2.append(a)
            peak2.append(b)
    L_list = np.array(L_list)
    peak = np.array(peak)
    eps_c = np.array(eps_c)
    peak2 = np.array(peak2)
    eps_c2 = np.array(eps_c2)

    plt.plot(eps_c, peak, "ks--", fillstyle="none")
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
        with open("susceptibility.dat", "w") as f:
            for i, L in enumerate(L_list):
                f.write("%d\t%.8f\t%.8f\n" % (L, eps_c[i], peak[i]))

    # Plot susceptibility peak and location against system size.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # plot susceptibility peak
    ax[0].loglog(L_list, peak, "o")
    # ax[0].loglog(L_list, peak2, "x")
    c = polyfit(np.log10(L_list), np.log10(peak), deg=1)
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

    # plot peak location
    ax[1].loglog(L_list, eps_c, "-o")
    ax[1].set_xlabel(r"$L$")
    ax[1].set_ylabel(r"${\rm peak\ location\ }\langle \epsilon_m\rangle^B_s$")
    plt.tight_layout()
    plt.show()
    plt.close()


def find_max_by_spline(xdata, ydata):
    f = interpolate.interp1d(xdata, ydata, kind="cubic")
    x = np.linspace(xdata.min(), xdata.max(), 2000)
    y = f(x)
    idxm = y.argmax()
    xm = x[idxm]
    ym = y[idxm]
    return xm, ym


if __name__ == "__main__":
    os.chdir("data")
    # distrubition(90, 500)
    # varied_sample_size(64, 200)
    average_type_B()
