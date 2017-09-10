import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl


def read():
    with open("susceptibility.dat") as f:
        lines = f.readlines()
        L1 = np.zeros(len(lines))
        eps1 = np.zeros_like(L1)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            L1[i] = float(s[0])
            eps1[i] = float(s[1])
    with open("polar_order.dat") as f:
        lines = f.readlines()
        L2 = np.zeros(len(lines))
        eps2 = np.zeros_like(L2)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            L2[i] = float(s[1])
            eps2[i] = float(s[0])
    with open("correlation_length.dat") as f:
        lines = f.readlines()
        L3 = np.zeros(len(lines))
        eps3 = np.zeros_like(L3)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            eps3[i] = float(s[0])
            L3[i] = 0.5 * (float(s[1]) + float(s[2]))
    return L1, eps1, L2, eps2, L3, eps3


def fit_exp(eps, l, beta=None, reverse=False):
    def fun1(x, xc, lnA, b):
        return lnA + b * (x - xc)**(-beta)

    def fun2(x, xc, lnA, b, beta):
        return lnA + b * (x - xc)**(-beta)

    def fun3(x, yc, lnA, b):
        return yc + (b / (x - lnA))**(1 / beta)

    def fun4(x, yc, lnA, b, beta):
        return yc + (b / (x - lnA))**(1 / beta)

    if not reverse:
        x = eps
        y = np.log(l)
        if beta is not None:
            p0 = [0.03, -1, 1.5]
            p_min = [0, -np.inf, 0]
            p_max = [0.05, y.min(), np.inf]
            popt, pcov = curve_fit(fun1, x, y, p0, bounds=(p_min, p_max))
        else:
            p0 = [0.03, -1, 1.5, 0.5]
            p_min = [0, -np.inf, 0, 0]
            p_max = [0.05, y.min(), np.inf, np.inf]
            popt, pcov = curve_fit(fun2, x, y, p0, bounds=(p_min, p_max))
    else:
        x = np.log(l)
        y = eps
        if beta is not None:
            p0 = [0.03, -1, 1.5]
            p_min = [0, -np.inf, 0]
            p_max = [0.05, x.min(), np.inf]
            popt, pcov = curve_fit(fun3, x, y, p0, bounds=(p_min, p_max))
        else:
            p0 = [0.03, -1, 1.5, 0.5]
            p_min = [0, -np.inf, 0, 0]
            p_max = [0.05, x.min(), np.inf, np.inf]
            popt, pcov = curve_fit(fun3, x, y, p0, bounds=(p_min, p_max))

    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_pow(eps, l, beta=None):
    def fun1(x, xc, lnA):
        return lnA - beta * np.log(x - xc)

    def fun2(x, xc, lnA, beta):
        return lnA - beta * np.log(x - xc)

    x = eps
    y = np.log(l)
    if beta is not None:
        p0 = [0.03, 0.01]
        p_min = [0, -np.inf]
        p_max = [0.05, np.inf]
        popt, pcov = curve_fit(fun1, x, y, p0, bounds=(p_min, p_max))
    else:
        p0 = [0.03, 0.01, 1]
        p_min = [0, -np.inf, 0]
        p_max = [0.05, np.inf, np.inf]
        popt, pcov = curve_fit(fun2, x, y, p0, bounds=(p_min, p_max))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def show_KT(nu):
    def gene_curve(popt, eps, L, label=None):
        line, = ax1.plot(eps, L, "o")
        if label is not None:
            line.set_label(label)
        xc, lnA, b = popt
        x = np.linspace(eps[0], eps[-1], 100)
        y = np.exp(lnA + b * (x - xc)**(-nu))
        ax1.plot(x, y, color=line.get_color())
        line, = ax2.plot(eps - xc, (np.log(L) - lnA) * (eps - xc)**(nu), "o")
        ax2.axhline(b, color=line.get_color())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    L1, eps1, L2, eps2, L3, eps3 = read()
    cellText = []
    popt, perr = fit_exp(eps1, L1, nu)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps1, L1, "from susceptibility peak")
    popt, perr = fit_exp(eps2, L2, nu)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps2, L2, "from order parameter")
    popt, perr = fit_exp(eps3, L3, nu)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps3, L3, "from correlation function")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel(r"$\xi$")
    ax1.set_yscale("log")
    ax1.legend()
    ax2.set_ylabel(r"$(\ln{\xi}-\ln{A_\xi})\cdot (\epsilon-\epsilon_c)^{\nu}$")
    ax2.set_xlabel(r"$\epsilon-\epsilon_c$")
    col_labels = [r"$\epsilon_c$", r"$\ln{A_{\xi}}$", r"$b$"]
    row_labels = [1, 2, 3]
    ax2.table(
        cellText=cellText,
        loc='center right',
        colLabels=col_labels,
        rowLabels=row_labels,
        colWidths=[0.3] * 3)
    plt.suptitle(r"${\rm KT-like\ scaling\ with\ } \nu=%g$" % nu)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def varied_nu():
    def cal_error_square(xdata, ydata, popt, nu):
        lny = popt[1] + popt[2] * (xdata - popt[0])**(-nu)
        return np.sum((np.log(ydata) - lny)**2)

    L1, eps1, L2, eps2, L3, eps3 = read()
    nus = np.linspace(0.01, 2, 50)
    # nus = np.logspace(np.log10(0.005), np.log10(3), 100)
    eps_c1 = np.zeros_like(nus)
    eps_c2 = np.zeros_like(nus)
    eps_c3 = np.zeros_like(nus)
    err1 = np.zeros_like(nus)
    err2 = np.zeros_like(nus)
    err3 = np.zeros_like(nus)
    lnA1 = np.zeros_like(nus)
    lnA2 = np.zeros_like(nus)
    lnA3 = np.zeros_like(nus)
    b1 = np.zeros_like(nus)
    b2 = np.zeros_like(nus)
    b3 = np.zeros_like(nus)
    e1 = np.zeros_like(nus)
    e2 = np.zeros_like(nus)
    e3 = np.zeros_like(nus)

    for i, nu in enumerate(nus):
        popt, perr = fit_exp(eps1, L1, nu)
        eps_c1[i], lnA1[i], b1[i] = popt
        err1[i] = perr[0]
        e1[i] = cal_error_square(eps1, L1, popt, nu)
        popt, perr = fit_exp(eps2, L2, nu)
        eps_c2[i], lnA2[i], b2[i] = popt
        err2[i] = perr[0]
        e2[i] = cal_error_square(eps2, L2, popt, nu)
        popt, perr = fit_exp(eps3, L3, nu)
        eps_c3[i], lnA3[i], b3[i] = popt
        e3[i] = cal_error_square(eps3, L3, popt, nu)
        err3[i] = perr[0]
    # plt.errorbar(nus, eps_c1, yerr=err1, fmt=".", label="from susceptibility peak")
    # plt.errorbar(nus, eps_c2, yerr=err2, fmt=".", label="from order parameter")
    # plt.errorbar(nus, eps_c3, yerr=err3, fmt=".", label="from correlation function")

    # plt.subplot(311)
    plt.plot(nus, eps_c1, "-", label="from susceptibility peak")
    plt.plot(nus, eps_c2, "-", label="from order parameter")
    plt.plot(nus, eps_c3, "-", label="from correlation function")
    plt.xscale("log")
    plt.legend()
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\epsilon_c$")

    # plt.subplot(312)
    # plt.plot(nus, lnA1, nus, lnA2, nus, lnA3)
    # plt.xscale("log")
    # plt.subplot(313)
    # plt.plot(nus, e1, nus, e2, nus, e3)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.tight_layout()
    plt.show()
    plt.close()


def show_algebraic():
    def gene_curve(popt, eps, L, label=None):
        line, = ax1.plot(eps, L, "o")
        if label is not None:
            line.set_label(label)
        xc, A, nu = popt
        x = np.linspace(eps[0], eps[-1], 100)
        y = A * (x - xc)**(-nu)
        ax1.plot(x, y, color=line.get_color())
        line, = ax2.plot(eps - xc, L * (eps - xc)**(nu), "o")
        ax2.axhline(A, color=line.get_color())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    L1, eps1, L2, eps2, L3, eps3 = read()
    cellText = []
    popt, perr = fit_pow(eps1, L1)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps1, L1, "from susceptibility peak")
    popt, perr = fit_pow(eps2, L2)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps2, L2, "from order parameter")
    popt, perr = fit_pow(eps3, L3)
    cellText.append([
        r"$%.4f\pm %.4f$" % (popt[0], perr[0]),
        r"$%.2f\pm %.2f$" % (popt[1], perr[1]),
        r"$%.3f\pm %.3f$" % (popt[2], perr[2])
    ])
    gene_curve(popt, eps3, L3, "from correlation function")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel(r"$\xi$")
    ax1.set_yscale("log")
    ax1.legend()
    ax2.set_ylabel(r"$\xi (\epsilon-\epsilon_c)^{\nu}$")
    ax2.set_xlabel(r"$\epsilon-\epsilon_c$")
    # col_labels = [r"$\epsilon_c$", r"$\ln{A_{\xi}}$", r"$b$"]
    # row_labels = [1, 2, 3]
    # ax2.table(
    #     cellText=cellText,
    #     loc='center right',
    #     colLabels=col_labels,
    #     rowLabels=row_labels,
    #     colWidths=[0.3] * 3)
    plt.suptitle(r"${\rm Algebraic\ scaling }$")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # L1, eps1, L2, eps2, L3, eps3 = read()
    # popt, perr = fit_exp(eps1, L1, 1)
    # print(popt, perr)
    # popt, perr = fit_exp(eps2, L2, 1)
    # print(popt, perr)
    # popt, perr = fit_exp(eps3, L3, 1)
    # print(popt, perr)
    varied_nu()
    # show_KT(0.01)
