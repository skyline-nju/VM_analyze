import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from numpy.polynomial.polynomial import polyfit


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
        if L.size > 2 and eps >= 0.055:
            dict_eps[eps] = {"L": L, "phi": phi}


def find_peak(alpha, show=True, output=False):
    dict_eps = {}
    files = glob.glob("0*.dat")
    for file in files:
        read(file, dict_eps)
    xm = np.zeros(len(dict_eps.keys()))
    ym = np.zeros_like(xm)
    eps_m = np.zeros_like(xm)
    for i, key in enumerate(dict_eps):
        L = dict_eps[key]["L"]
        phi = dict_eps[key]["phi"]
        Y = phi * L**alpha
        if show:
            line, = plt.plot(L, Y, "o", label="%.4f" % key)

        c = polyfit(np.log10(L), np.log10(Y), 5)
        x = np.linspace(np.log10(L[0]), np.log10(L[-1]), 10000)
        y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4 + c[5] * x**5
        x = 10**x
        y = 10**y
        idxm = y.argmax()
        xm[i] = x[idxm]
        ym[i] = y[idxm]
        eps_m[i] = key
        if show:
            plt.plot(x, y, color=line.get_color())

    c, stats = polyfit(np.log10(xm), np.log10(ym), 1, full=True)
    x = np.linspace(np.log10(xm[0]) + 0.05, np.log10(xm[-1]) - 0.05, 1000)
    y = c[0] + c[1] * x
    if output:
        with open("polar_order.dat", "w") as f:
            for i in range(eps_m.size):
                f.write("%.4f\t%.8f\n" % (eps_m[i], xm[i]))
    if show:
        plt.plot(xm, ym, "ks", fillstyle="none")
        plt.plot(10**x, 10**y, "k--", label=r"$L^{%.3f}$" % (c[1]))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$L$")
        plt.ylabel(r"$L^{%g}\phi$" % alpha)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        return c[1], stats[0][0], eps_m, xm


def varied_alpha():
    alpha = np.linspace(0.4, 0.55, 300)
    beta = np.zeros_like(alpha)
    res = np.zeros_like(alpha)
    for i in range(beta.size):
        beta[i], res[i], eps_m, Lm = find_peak(alpha[i], False)
    plt.figure(figsize=(8, 4))
    c = polyfit(alpha, beta, 1)
    plt.subplot(121)
    plt.plot(alpha, beta, label=r"$\beta=%.4f+%.4f\alpha$" % (c[0], c[1]))
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.legend()
    plt.subplot(122)
    plt.plot(alpha, res)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data")
    # varied_alpha()
    find_peak(0.46)
