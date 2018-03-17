'''
Fit data points (epsilon, correlation length) with KT-like scaling or algebraic
scaling.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def read(eta, head1=0, tail1=0, head2=0, tail2=0):
    """ Read three set of (epsilon, correlation length) from three files,
        respectively.

    Parameters:
    --------
    head1: int, optional
        Remove the first `head1` lines from file1.
    tail1: int, optional
        Remove the last 'tail1' lines from file1.
    head2: int, optional
        Remove the first `head2` lines from file2.
    tail2: int, optional
        Remove the last `tail2` lines from file2.
    """
    path = r"data\eta=%.2f" % eta
    from suscept_peak import read_suscept_peak
    L1, eps1 = read_suscept_peak(eta, head1, tail1)
    print("--------")
    with open(path + os.path.sep + "polar_order.dat") as f:
        lines = f.readlines()
        n = len(lines)
        lines = lines[head2:n - tail2]
        L2 = np.zeros(len(lines))
        eps2 = np.zeros_like(L2)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            L2[i] = float(s[1])
            eps2[i] = float(s[0])
            print(eps2[i], L2[i])
    if eta == 0.1:
        return L1, eps1, L2, eps2
    else:
        with open(path + os.path.sep + "correlation_length.dat") as f:
            lines = f.readlines()
            L3 = np.zeros(len(lines))
            eps3 = np.zeros_like(L3)
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                eps3[i] = float(s[0])
                if len(s) == 3:
                    L3[i] = 0.5 * (float(s[1]) + float(s[2]))
                else:
                    L3[i] = float(s[1])
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


def plot_KT_fit(nu, ax, eps, xi, reversed=False, eps_min=0.05, eps_max=0.087):
    popt, perr = fit_exp(eps, xi, beta=nu)
    x = np.linspace(eps_min, eps_max, 100)
    y = np.exp(popt[1] + popt[2] * (x - popt[0])**(-nu))
    label = r"$\xi=%.3f \times e^{%.3f(\epsilon - %.4f)^{-%g}}$" % (
        np.exp(popt[1]), popt[2], popt[0], nu)
    if reversed:
        ax.plot(y, x, "--", label=label)
    else:
        ax.plot(x, y, "--", label=label)


def plot_pow_fit(ax, eps, xi, reversed=False, eps_min=0.05, eps_max=0.087):
    popt, perr = fit_pow(eps, xi)
    x = np.linspace(eps_min, eps_max, 100)
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
    return None, None


def varied_nu3(head1=0, tail1=0, head2=1, tail2=0):
    """
    Epsilon evaluated from three different correlation lengths vs. the
    exponent nu.
    """
    L1, eps1, L2, eps2, L3, eps3 = read(head1, tail1, head2, tail2)
    nus = np.linspace(0.05, 2, 200)
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


def varied_nu2(eta,
               head1=0,
               tail1=0,
               head2=1,
               tail2=0,
               save_fig=False,
               alpha=None,
               KT_scaling=True):
    """
    Epsilon evaluated from two different correlation lengths vs. the
    exponent nu.
    """
    if eta == 0.18:
        L1, eps1, L2, eps2, L3, eps3 = read(eta, head1, tail1, head2, tail2)
    else:
        L1, eps1, L2, eps2 = read(eta, head1, tail1, head2, tail2)
    if alpha is not None:
        from corr_len import get_phi_dict, find_peak2
        phi_dict = get_phi_dict(eta)
        L2, ym, eps2 = find_peak2(phi_dict, alpha)
        L2 = L2[head2:L2.size - tail2]
        eps2 = eps2[head2:eps2.size - tail2]

    if KT_scaling:
        nu_arr = np.linspace(0.1, 3, 200)
    else:
        nu_arr = np.linspace(1, 3, 100)
    eps_c1 = np.zeros_like(nu_arr)
    eps_c2 = np.zeros_like(nu_arr)

    for i, nu in enumerate(nu_arr):
        try:
            if KT_scaling:
                popt, perr = fit_exp(eps1, L1, nu)
                eps_c1[i], lnA, b = popt
                popt, perr = fit_exp(eps2, L2, nu)
                eps_c2[i], lnA, b = popt
            else:
                popt, perr = fit_pow(eps1, L1, nu)
                eps_c1[i], lnA = popt
                popt, perr = fit_pow(eps2, L2, nu)
                eps_c2[i], lnA = popt
        except:
            eps_c1[i] = np.nan
            eps_c2[i] = np.nan
            print(nu)

    fig = plt.figure()
    if KT_scaling:
        lb1 = r"KT-like scaling for $\epsilon^L_m(L), L=%d,%d,\cdots,%d$"
        lb2 = r"KT-like scaling for $\xi(\epsilon),\epsilon = %g,%g\cdots,%g$"
    else:
        lb1 = r"algebraic scaling for $\epsilon^L_m(L), L=%d,%d,\cdots,%d$"
        lb2 = r"algebraic scaling for $\xi(\epsilon),\epsilon=%g,%g\cdots,%g$"
    plt.plot(nu_arr, eps_c1, "-", label=lb1 % (L1[0], L1[1], L1[-1]))
    plt.plot(nu_arr, eps_c2, "-", label=lb2 % (eps2[0], eps2[1], eps2[-1]))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\nu$", fontsize="x-large")
    plt.ylabel(r"$\epsilon_c$", fontsize="x-large")
    plt.tight_layout()
    plt.legend(loc="best")

    xc, yc = get_cross_point(nu_arr, eps_c1, eps_c2)
    if xc is not None:
        ax = fig.add_axes([0.35, 0.38, 0.4, 0.4])
        ax.plot(nu_arr, eps_c1, "-")
        ax.plot(nu_arr, eps_c2, "-")
        ax.axhline(yc, c="r", linestyle="dashed")
        ax.axvline(xc, c="r", linestyle="dashed")
        ax.set_xlim(xc - 0.02, xc + 0.02)
        ax.set_ylim(yc - 0.0003, yc + 0.0003)
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel(r"$\epsilon_c$")
    if save_fig:
        plt.savefig("KT_nu2_%d_%d_%d_%d.eps" % (head1, tail1, head2, tail2))
    else:
        plt.show()
    plt.close()


def cross_point_w_varied_alpha(eta,
                               h1=0,
                               t1=0,
                               h2=0,
                               t2=0,
                               save_fig=False,
                               KT_scaling=True):
    """ Get cross point of epsilon_c vs. nu curves for the susceptibility peak
        and correlation length, respectively.
    """
    from suscept_peak import read_suscept_peak
    from corr_len import get_phi_dict, find_peak2
    L1, eps1 = read_suscept_peak(eta, h1, t1)
    if eta == 0.18:
        if KT_scaling:
            alpha_arr = np.linspace(0.45, 0.7, 26)
        else:
            alpha_arr = np.linspace(0.45, 0.8, 36)
    else:
        if KT_scaling:
            alpha_arr = np.linspace(0.6, 0.72, 25)
        else:
            alpha_arr = np.linspace(0.5, 0.85, 40)
    phi_dict = get_phi_dict(eta)

    if KT_scaling:
        nu_arr = np.linspace(0.1, 3, 150)
        f_pat = r"data\varied_alpha_eta=%g_%d_%d_%d_%d.dat"
    else:
        nu_arr = np.linspace(1, 2, 80)
        f_pat = r"data\varied_alpha_pow_eta=%g_%d_%d_%d_%d.dat"
    filename = f_pat % (eta, h1, t1, h2, t2)
    eps_c_arr = np.zeros_like(alpha_arr)
    nu_c_arr = np.zeros_like(alpha_arr)
    eps_c1 = np.zeros_like(nu_arr)
    for j, nu in enumerate(nu_arr):
        if KT_scaling:
            popt, perr = fit_exp(eps1, L1, nu)
            eps_c1[j], lnA, b = popt
        else:
            popt, perr = fit_pow(eps1, L1, nu)
            eps_c1[j], lnA = popt

    for i, alpha in enumerate(alpha_arr):
        try:
            eps_c2 = np.zeros_like(nu_arr)
            for j, nu in enumerate(nu_arr):
                L2, ym, eps2 = find_peak2(phi_dict, alpha)
                if KT_scaling:
                    popt, perr = fit_exp(eps2[h2:eps2.size - t2],
                                         L2[h2:L2.size - t2], nu)
                    eps_c2[j], lnA, b = popt
                else:
                    popt, perr = fit_pow(eps2[h2:eps2.size - t2],
                                         L2[h2:L2.size - t2], nu)
                    eps_c2[j], lnA = popt
            nu_c_arr[i], eps_c_arr[i] = get_cross_point(nu_arr, eps_c1, eps_c2)
            print("success for i =", i, "alpha =", alpha)
        except:
            nu_c_arr[i] = np.nan
            eps_c_arr[i] = np.nan
    with open(filename, "w") as f:
        for i, alpha in enumerate(alpha_arr):
            f.write("%f\t%f\t%f\n" % (alpha, nu_c_arr[i], eps_c_arr[i]))


def plot_nu_and_eps_c_vs_alpha():
    def read_data(eta):
        with open(r"data\varied_alpha_eta=%g.dat" % eta) as f:
            lines = f.readlines()
            alpha, nu, eps_c = np.zeros((3, len(lines)))
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                alpha[i] = float(s[0])
                nu[i] = float(s[1])
                eps_c[i] = float(s[2])
        return alpha, nu, eps_c

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for eta in [0.1, 0.18]:
        alpha, nu, eps_c = read_data(eta)
        ax1.plot(alpha, nu, "-o", label=r"$\eta=%.2f$" % eta)
        ax2.plot(alpha, eps_c, "-s", label=r"$\eta=%.2f$" % eta)
        ax3.plot(eps_c, nu, ">", label=r"$\eta=%.2f$" % eta)
    ax1.set_xlabel(r"$\alpha$", fontsize="x-large")
    ax2.set_xlabel(r"$\alpha$", fontsize="x-large")
    ax1.set_ylabel(r"$\nu^*$", fontsize="x-large")
    ax2.set_ylabel(r"$\epsilon_c^*$", fontsize="x-large")
    ax3.set_xlabel(r"$\epsilon_c^*$", fontsize="x-large")
    ax3.set_ylabel(r"$\nu^*$", fontsize="x-large")

    ax1.legend(fontsize="large")
    ax2.legend(fontsize="large")
    ax3.legend(fontsize="large")
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


def plot_cross_point(eta, KT_scaling=True):
    def read_data(infile):
        with open(infile) as f:
            lines = f.readlines()
            alpha, nu, eps_c = np.zeros((3, len(lines)))
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                alpha[i] = float(s[0])
                nu[i] = float(s[1])
                eps_c[i] = float(s[2])
        return alpha, nu, eps_c

    if KT_scaling:
        f_pre = r"data/varied_alpha_eta="
    else:
        f_pre = r"data/varied_alpha_pow_eta="
    if (eta == 0.10):
        alpha1, nu1, eps_c1 = read_data(f_pre + r"%g_3_0_0_0.dat" % eta)
        alpha2, nu2, eps_c2 = read_data(f_pre + r"%g_3_0_3_0.dat" % eta)
        alpha3, nu3, eps_c3 = read_data(f_pre + r"%g_3_0_0_3.dat" % eta)
    else:
        alpha1, nu1, eps_c1 = read_data(f_pre + r"%g_1_1_0_0.dat" % eta)
        alpha2, nu2, eps_c2 = read_data(f_pre + r"%g_1_1_3_0.dat" % eta)
        alpha3, nu3, eps_c3 = read_data(f_pre + r"%g_1_1_0_3.dat" % eta)

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
    ax1.plot(alpha1, nu1, "o")
    ax1.plot(alpha2, nu2, "s")
    ax1.plot(alpha3, nu3, "<")
    ax2.plot(alpha1, eps_c1, "o")
    ax2.plot(alpha2, eps_c2, "s")
    ax2.plot(alpha3, eps_c3, "<")
    ax1.set_xlabel(r"$\alpha$", fontsize="x-large")
    ax2.set_xlabel(r"$\alpha$", fontsize="x-large")
    ax1.set_ylabel(r"$\nu^*$", fontsize="x-large")
    ax2.set_ylabel(r"$\epsilon^*_c$", fontsize="x-large")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.suptitle(r"$\eta=%g$" % eta, fontsize="x-large", y=0.995)
    plt.show()
    plt.close()


if __name__ == "__main__":
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
    eta = 0.1
    # varied_nu2(0.18, 1, 1, 0, 0, False, 0.6, KT_scaling=True)
    # cross_point_w_varied_alpha(0.1, 3, 0, 3, 0, False, KT_scaling=False)
    plot_cross_point(0.1, KT_scaling=False)
    # plot_nu_and_eps_c_vs_alpha()
    # varied_nu3()
    # show_KT(1)
    # show_algebraic()
