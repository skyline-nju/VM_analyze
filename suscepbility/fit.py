import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def read():
    with open("suscept_peak.dat") as f:
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


def fit_exp(eps, l, beta=None, reverse=False, ret_res=False):
    def fun1(x, xc, lnA, b):
        return lnA + b * (x - xc)**(-beta)

    def fun2(x, xc, lnA, b, beta):
        return lnA + b * (x - xc)**(-beta)

    def fun3(x, yc, lnA, b):
        return yc + (b / (x - lnA))**(1 / beta)

    def fun4(x, yc, lnA, b, beta):
        return yc + (b / (x - lnA))**(1 / beta)

    def cal_squared_residuals(f, xdata, ydata, popt):
        yfit = np.zeros_like(ydata)
        for i in range(xdata.size):
            if beta is not None:
                yfit[i] = f(xdata[i], popt[0], popt[1], popt[2])
            else:
                yfit[i] = f(xdata[i], popt[0], popt[1], popt[2], popt[3])
        squared_res = np.var(yfit - ydata)
        return squared_res

    if not reverse:
        x = eps
        y = np.log(l)
        if beta is not None:
            p0 = [0.03, -1, 1.5]
            p_min = [0, -np.inf, 0]
            p_max = [0.05, y.min(), np.inf]
            popt, pcov = curve_fit(fun1, x, y, p0, bounds=(p_min, p_max))
            if ret_res:
                squared_res = cal_squared_residuals(fun1, x, y, popt)
        else:
            p0 = [0.03, -1, 1.5, 0.5]
            p_min = [0, -np.inf, 0, 0]
            p_max = [0.05, y.min(), np.inf, np.inf]
            popt, pcov = curve_fit(fun2, x, y, p0, bounds=(p_min, p_max))
            if ret_res:
                squared_res = cal_squared_residuals(fun2, x, y, popt)
    else:
        x = np.log(l)
        y = eps
        if beta is not None:
            p0 = [0.03, -1, 1.5]
            p_min = [0, -np.inf, 0]
            p_max = [0.05, x.min(), np.inf]
            popt, pcov = curve_fit(fun3, x, y, p0, bounds=(p_min, p_max))
            if ret_res:
                squared_res = cal_squared_residuals(fun3, x, y, popt)
        else:
            p0 = [0.03, -1, 1.5, 0.5]
            p_min = [0, -np.inf, 0, 0]
            p_max = [0.05, x.min(), np.inf, np.inf]
            popt, pcov = curve_fit(fun4, x, y, p0, bounds=(p_min, p_max))
            if ret_res:
                squared_res = cal_squared_residuals(fun4, x, y, popt)

    perr = np.sqrt(np.diag(pcov))
    if ret_res:
        return popt, perr, squared_res
    else:
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


def plot_KT_fit(nu, ax, eps, xi, reversed=False):
    popt, perr = fit_exp(eps, xi, beta=nu)
    x = np.linspace(0.05, 0.087, 100)
    y = np.exp(popt[1] + popt[2] * (x - popt[0])**(-nu))
    label = r"$\xi=%.3f \times e^{%.3f(\epsilon - %.4f)^{-%g}}$" % (
        np.exp(popt[1]), popt[2], popt[0], nu)
    if reversed:
        ax.plot(y, x, "--", label=label)
    else:
        ax.plot(x, y, "--", label=label)


def plot_pow_fit(ax, eps, xi, reversed=False):
    popt, perr = fit_pow(eps, xi)
    x = np.linspace(0.05, 0.087, 100)
    y = np.exp(popt[1] - popt[2] * np.log(x - popt[0]))
    label = r"$\xi=%.3f \times (\epsilon-%.4f)^{-%.3f}$" % (np.exp(popt[1]),
                                                            popt[0], popt[2])
    if reversed:
        ax.plot(y, x, "--", label=label)
    else:
        ax.plot(x, y, "--", label=label)


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


def get_cross_point(x, y1, y2):
    for i in range(x.size - 1):
        if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) < 0:
            k1 = (y1[i] - y1[i + 1]) / (x[i] - x[i + 1])
            k2 = (y2[i] - y2[i + 1]) / (x[i] - x[i + 1])
            xc = x[i] - (y1[i] - y2[i]) / (k1 - k2)
            yc = y1[i] - (x[i] - xc) * k1
            return xc, yc
    return None


def varied_nu():
    L1, eps1, L2, eps2, L3, eps3 = read()
    nus = np.linspace(0.05, 2, 300)
    # nus = np.logspace(np.log10(0.005), np.log10(3), 100)
    eps_c1, eps_c2, eps_c3 = np.zeros((3, nus.size))
    err1, err2, err3 = np.zeros((3, nus.size))
    lnA1, lnA2, lnA3 = np.zeros((3, nus.size))
    b1, b2, b3 = np.zeros((3, nus.size))

    for i, nu in enumerate(nus):
        popt, perr = fit_exp(eps1, L1, nu)
        eps_c1[i], lnA1[i], b1[i] = popt
        popt, perr = fit_exp(eps2, L2, nu)
        eps_c2[i], lnA2[i], b2[i] = popt
        popt, perr = fit_exp(eps3, L3, nu)
        eps_c3[i], lnA3[i], b3[i] = popt

    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax1.plot(nus, eps_c1, "-", label="from susceptibility peak")
    ax1.plot(nus, eps_c2, "-", label="from order parameter")
    ax1.plot(nus, eps_c3, "-", label="from correlation function")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\nu$", fontsize="x-large")
    ax1.set_ylabel(r"$\epsilon_c$", fontsize="x-large")

    y = np.array([eps_c1, eps_c2, eps_c3])
    y_std = np.std(y, axis=0)
    y_mean = np.mean(y, axis=0)
    ax2.plot(nus, y_std / y_mean, "k")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\nu$", fontsize="x-large")
    ylabel = r"$\left.\sqrt{\langle (\epsilon_c - \overline{\epsilon_c})^2" \
        + r"\rangle} \middle / \overline{\epsilon_c} \right.$"
    ax2.set_ylabel(ylabel, fontsize="x-large")
    idx = np.argmin(y_std / y_mean)
    nu_m = nus[idx]
    eps_m = y_mean[idx]

    ax1.axhline(eps_m, color="r", linestyle="--")
    ax1.axvline(nu_m, color="r", linestyle="--")
    ax2.axvline(
        nu_m, color="r", linestyle="--", label=r"optimal $\nu=%.4f$" % (nu_m))
    ax1.legend(fontsize="large")
    ax2.legend(fontsize="large")
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
    os.chdir("data")
    # L1, eps1, L2, eps2, L3, eps3 = read()
    # popt, perr = fit_exp(eps1, L1, 1)
    # print(popt, perr)
    # popt, perr = fit_exp(eps2, L2, 1)
    # print(popt, perr)
    # popt, perr = fit_exp(eps3, L3, 1)
    # print(popt, perr)
    # popt, perr = fit_pow(eps1, L1)
    # print(popt, perr)
    # popt, perr = fit_pow(eps2, L2)
    # print(popt, perr)
    # popt, perr = fit_pow(eps3, L3)
    # print(popt, perr)
    varied_nu()
    # show_KT(0.01)
