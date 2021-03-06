"""
Estimate the correlation length beyond which the order parameter begin to decay
with a power law.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.polynomial.polynomial import polyfit
from fit import plot_KT_fit, plot_pow_fit, fit_pow, find_peak_polyfit
import sys
sys.path.append("..")
try:
    from corr2d.add_line import add_line
except ImportError:
    print("error when import add_line")


def get_phi_dict(eta, eps_min=None, disorder_t="RT"):
    """
    Get dict of phi with key `eps`. phi_dict[eps] is a 2 * n array,
    L_arr=phi_dict[eps][0], phi_arr = phi_dict[eps][1].
    """
    from create_dict import create_dict_from_xlsx
    if disorder_t == "RT":
        path = r"D:\data\VM2d\random_torque\sample_average"
    elif disorder_t == "RF":
        path = r"D:\data\VM2d\random_field\sample_average"

    infile = path + os.path.sep + r"eta=%g.xlsx" % eta
    if disorder_t == "RT":
        if eta == 0.18:
            # from create_dict import create_dict_from_txt
            # path = r"data\eta=%.2f" % eta
            if eps_min is None:
                eps_min = 0.052
            L_min = None
            phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min,
                                             L_min, "dict-arr", 7)
        elif eta == 0.1:
            # from create_dict import create_dict_from_xlsx
            # path = r"E:\data\random_torque\susceptibility"
            # infile = path + os.path.sep + r"eta=%g.xlsx" % eta
            if eps_min is None:
                eps_min = 0.048
            L_min = None
            phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min,
                                             L_min, "dict-arr", 5)
            del phi_dict[0.053]
            del phi_dict[0.055]
            # del phi_dict[0.048]
        elif eta == 0.05:
            if eps_min is None:
                eps_min = 0.03
            L_min = None
            phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min,
                                             L_min, "dict-arr", 4)
    elif disorder_t == "RF":
        if eta == 0.18:
            eps_min = None
            L_min = None
            phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min,
                                             L_min, "dict-arr", 4)
    return phi_dict


def plot_phi_vs_L(phi_dict,
                  ax=None,
                  eta=None,
                  Lc=None,
                  phi_c=None,
                  eps_max=None):
    """ PLot phi against L with varied epsilon in log-log scales."""
    if ax is None:
        ax = plt.gca()
        flag_show = True
    else:
        flag_show = False
    eps_valid = []
    if eps_max is not None:
        for key in sorted(phi_dict.keys()):
            if key <= eps_max:
                eps_valid.append(key)
    else:
        eps_valid = [key for key in sorted(phi_dict.keys())]
    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(eps_valid)))
    for i, eps in enumerate(eps_valid):
        L, phi = phi_dict[eps]
        # x = L / L ** (1 - eps)
        ax.plot(L, phi, "-o", label="%.4f" % eps, color=color[i])
    if Lc is not None and phi_c is not None:
        ax.plot(Lc, phi_c, "--ks", fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$", fontsize="x-large")
    ax.set_ylabel(r"$\phi$", fontsize="x-large")
    if eta == 0.05:
        ax.set_xlim(45)
        ax.set_ylim(ymax=0.9)
        add_line(
            ax,
            0.5,
            0.86,
            0.95,
            -0.051,
            scale="log",
            label=r"$-0.051$",
            yl=0.85)
        add_line(
            ax,
            0.5,
            0.74,
            0.95,
            -0.107,
            scale="log",
            label=r"$-0.107$",
            yl=0.7)
        add_line(
            ax,
            0.5,
            0.65,
            0.95,
            -0.167,
            scale="log",
            label=r"$-0.167$",
            yl=0.35,
            xl=0.8)
    ax.legend(title=r"$\epsilon=$", loc="lower left", fontsize="xx-large")
    ax.set_title(r"$\eta=%g,\ \rho_0=1$" % eta, fontsize="xx-large")
    # add_line(ax, 0.25, 0.65, 0.55, -1, scale="log")
    if flag_show:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_slope_vs_L(phi_dict,
                    ax=None,
                    eta=None,
                    Lc=None,
                    phi_c=None,
                    eps_max=None):
    if ax is None:
        ax = plt.gca()
        flag_show = True
    else:
        flag_show = False
    eps_valid = []
    if eps_max is not None:
        for key in sorted(phi_dict.keys()):
            if key <= eps_max:
                eps_valid.append(key)
    else:
        eps_valid = [key for key in sorted(phi_dict.keys())]
    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(eps_valid)))
    for i, eps in enumerate(eps_valid):
        print("eps =", eps)
        L, phi = phi_dict[eps]
        x = np.sqrt(L[1:] * L[:-1])
        y = -np.log10(phi[1:] / phi[:-1]) / np.log10(L[1:] / L[:-1])
        for j in range(x.size):
            print(x[j], y[j])
        ax.plot(x, y, "-o", label="%.4f" % eps, color=color[i])
    if Lc is not None and phi_c is not None:
        ax.plot(Lc, phi_c, "--ks", fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$", fontsize="x-large")
    ax.set_ylabel(r"$\phi$", fontsize="x-large")
    ax.legend(title=r"$\epsilon=$", loc="lower left", fontsize="xx-large")
    ax.set_title(r"$\eta=%g,\ \rho_0=1$" % eta, fontsize="xx-large")
    # add_line(ax, 0.25, 0.65, 0.55, -1, scale="log")
    if flag_show:
        plt.tight_layout()
        plt.show()
        plt.close()


def collapse_phi_L(eta,
                   phi_dict,
                   eps_max=None,
                   flag_ylim=False,
                   square_eps=True,
                   ax=None):
    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        flag_show = True
    else:
        flag_show = False
    eps_valid = []
    if eps_max is not None:
        for key in sorted(phi_dict.keys()):
            if key <= eps_max and key != 0.001 \
                    and key != 0.048 and key != 0.049:
                eps_valid.append(key)
    else:
        eps_valid = [key for key in sorted(phi_dict.keys())]
    eps_valid = np.array(eps_valid)
    if square_eps:
        alpha = 1 - eps_valid**2
        # gamma_eq = r"\gamma(\epsilon) = 1 - \epsilon ^ 2"
    else:
        alpha = 1 - eps_valid
        # gamma_eq = r"\gamma(\epsilon) = 1 - \epsilon"
    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(eps_valid)))
    for i, eps in enumerate(eps_valid):
        L, phi = phi_dict[eps]
        ax.plot(
            L / L**alpha[i],
            phi,
            "o",
            fillstyle="none",
            label=r"$\epsilon=%g$" % eps,
            color=color[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L/L^{\gamma (\epsilon)}$", fontsize="x-large")
    ax.set_ylabel(r"$\Phi$", fontsize="x-large")
    if flag_ylim:
        ax.set_ylim(0.06, 1.1)

    ax.legend(title=r"$\eta=%g$" % (eta))
    if flag_show:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_collapse_phi_L(square_eps=True):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    phi_dict_10 = get_phi_dict(0.1, 0)
    phi_dict_18 = get_phi_dict(0.18, 0)
    eps_max = 0.05
    collapse_phi_L(0.1, phi_dict_10, eps_max, False, square_eps, axes[0])
    collapse_phi_L(0.18, phi_dict_18, eps_max, False, square_eps, axes[1])
    collapse_phi_L(0.1, phi_dict_10, eps_max, True, square_eps, axes[2])
    collapse_phi_L(0.18, phi_dict_18, eps_max, True, square_eps, axes[3])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if square_eps:
        title = r"$\gamma (\epsilon) = 1 - \epsilon ^ 2$"
    else:
        title = r"$\gamma (\epsilon) = 1 - \epsilon$"
    plt.suptitle(title, fontsize="xx-large", y=0.99)
    plt.show()
    plt.close()


def find_peak(eta,
              phi_dict,
              alpha,
              show=True,
              save_data=False,
              ax=None,
              ret=False):
    """ Find the peak of phi * L ** alpha against L in log-log scales."""
    if show:
        if ax is None:
            ax = plt.gca()
            flag_show = True
        else:
            flag_show = False
    xm = np.zeros(len(phi_dict.keys()))
    ym = np.zeros_like(xm)
    eps_m = np.zeros_like(xm)
    phi_m = np.zeros_like(xm)
    xerr = np.zeros_like(xm)
    yerr = np.zeros_like(xm)
    color = plt.cm.gist_rainbow(np.linspace(0, 1, xm.size))
    for i, key in enumerate(sorted(phi_dict.keys())):
        L = phi_dict[key][0]
        phi = phi_dict[key][1]
        Y = phi * L**alpha
        xm[i], ym[i], xerr[i], yerr[i], c = find_peak_polyfit(
            L, Y, "log", "log")
        phi_m[i] = ym[i] / xm[i]**alpha
        eps_m[i] = key
        print("%g, x=%f±%f, y=%f±%f" % (key, xm[i], xerr[i], ym[i], yerr[i]))
        if show:
            x = np.linspace(L.min(), L.max(), 1000)
            y = np.exp(np.polyval(c, np.log(x)))
            ax.plot(L, Y, "o", label="%.4f" % key, color=color[i])
            ax.plot(x, y, color=color[i])

    c, stats = polyfit(np.log(xm), np.log(ym), 1, full=True)
    x = np.linspace(np.log(xm[0]) + 0.05, np.log(xm[-1]) - 0.05, 1000)
    y = c[0] + c[1] * x
    if save_data:
        with open(r"data\eta=%.2f\polar_order.dat" % eta, "w") as f:
            for i in range(eps_m.size):
                f.write("%.4f\t%.8f\n" % (eps_m[i], xm[i]))
    if show:
        ax.plot(xm, ym, "ks", fillstyle="none")
        ax.errorbar(xm, ym, xerr=xerr, yerr=yerr, fmt="none")
        ax.plot(np.exp(x), np.exp(y), "k--", label=r"$L^{%.3f}$" % (c[1]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$L$", fontsize="x-large")
        ax.set_ylabel(r"$L^{%g}\phi$" % alpha, fontsize="x-large")
        # ax.legend()
        if flag_show:
            plt.tight_layout()
            plt.show()
            plt.close()
    if ret:
        return c[1], stats[0][0], eps_m, xm, phi_m, xerr


def plot_three_panel(eta, alpha, save_fig=False, save_data=False):
    """ Plot phi vs. L, L^alpha * phi vs. L and correlation length vs. eps."""
    phi_dict = get_phi_dict(eta)
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)
    c0, res, eps_m, Lm, phi_m, xi_err = find_peak(eta, phi_dict, alpha, True,
                                                  save_data, ax2, True)
    plot_phi_vs_L(phi_dict, ax1, eta, Lm, phi_m)

    for i in range(Lm.size):
        print(Lm[i], phi_m[i] * Lm[i]**alpha, eps_m[i])
    ax3.plot(eps_m, Lm, "o")
    ax3.errorbar(eps_m, Lm, yerr=xi_err, fmt="none")

    if eta == 0.18:
        eps_min = 0.05
        eps_max = 0.087
    elif eta == 0.1:
        eps_min = 0.047
        eps_max = 0.076
    xi_min = 100
    mask = Lm >= xi_min
    x = eps_m[mask]
    y = Lm[mask]

    ax3.axhspan(y[-1] - 10, y[0] + 100, alpha=0.2)
    plot_KT_fit(0.5, ax3, x, y, eps_min=eps_min, eps_max=eps_max)
    plot_KT_fit(1.0, ax3, x, y, eps_min=eps_min, eps_max=eps_max)
    plot_pow_fit(ax3, x, y)

    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\epsilon$", fontsize="x-large")
    ax3.set_ylabel(r"$\xi$", fontsize="x-large")
    ax3.legend(fontsize="large", title="fitting curve")
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    ax3.set_title("(c)")
    if save_fig:
        plt.savefig(r"data\polar_order_eta%g.eps" % (eta * 100))
    else:
        plt.show()
    plt.close()


def collapse(ax,
             eta,
             phi_dict,
             text_pos,
             beta_over_nu,
             eps_c,
             nu,
             A=None,
             out=False,
             disorder_t="RT"):
    if disorder_t == "RT":
        if eta == 0.18:
            eps_max = 0.07
        elif eta == 0.1:
            eps_max = 0.057
    elif disorder_t == "RF":
        eps_max = 0.22
    for eps in sorted(phi_dict.keys()):
        if eps < eps_max:
            L_arr, phi_arr = phi_dict[eps]
            y = phi_arr * L_arr**(beta_over_nu)
            if A is None:
                x = L_arr**(1 / nu) * np.absolute(eps - eps_c)
                # x = np.log(L_arr ** (1/nu)) * np.absolute(eps - eps_c)
            else:
                x = (np.log(L_arr) - np.log(A))**(
                    1 / nu) * np.absolute(eps - eps_c)
            if eps < eps_c:
                ax.plot(x, y, "s", ms=5, label="%.3f" % eps, fillstyle="none")
                if out:
                    for i in range(x.size):
                        print(x[i], y[i])
            else:
                ax.plot(x, y, "o", ms=5, label="%.3f" % eps, fillstyle="none")
    text = r"$\nu=%g$" % nu + "\n" + r"$\epsilon_c=%g$" % eps_c
    if A is None:
        xlabel = r"$|\epsilon-\epsilon_c| L ^ {1/\nu}$"
    else:
        xlabel = r"$|\epsilon-\epsilon_c|\left[\ln L-\ln A_\xi\right]^{1/\nu}$"
        text += "\n" + r"$A_\xi=%g$" % A
    ax.set_xlabel(xlabel, fontsize="x-large")
    ax.text(
        text_pos[0],
        text_pos[1],
        text,
        transform=ax.transAxes,
        fontsize="x-large")
    ax.legend(
        title=r"$\epsilon=$",
        loc="best",
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.05)
    # ax.set_xscale("log")
    # ax.set_yscale("log")


def collapse3(eta, beta_over_nu=0.05, disorder_t="RT"):
    phi_dict = get_phi_dict(eta, 0.01, disorder_t=disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            eps_c = [0.045342993, 0.040390798076, 0.035436731]
            nu = [1.658079, 0.5, 1]
            A = [None, 1.938241, 10.16859853]
        elif eta == 0.1:
            eps_c = [0.040701, 0.035894968, 0.031156659]
            nu = [1.73105, 0.5, 1]
            A = [None, 1.55257517, 8.86459331]
    else:
        if eta == 0.18:
            eps_c = [0.102123839, 0.07033820, 0.04359504]
            nu = [2.122325, 0.5, 1]
            A = [None, 0.1670940, 1.727417]
    text_pos = [(0.1, 0.6), (0.1, 0.6), (0.1, 0.6)]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4.5),
        constrained_layout=True,
        sharey=True)
    for i, ax in enumerate(axes):
        if i == 0:
            out = True
        else:
            out = False
        collapse(ax, eta, phi_dict, text_pos[i], beta_over_nu, eps_c[i], nu[i],
                 A[i], out, disorder_t)

    if disorder_t == "RT":
        axes[0].set_xlim(1e-2)
        axes[1].set_xlim(4e-3)
    axes[0].set_ylabel(r"$\phi L^{\beta/\nu}$", fontsize="x-large")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")

    # slope1 = beta_over_nu * nu[0]
    # add_line(
    #     axes[0],
    #     0.5,
    #     0.82,
    #     1,
    #     slope1,
    #     scale="log",
    #     # label=r"slope = $\beta$",
    #     label=None,
    #     xl=0.65,
    #     yl=0.85)
    # slope2 = slope1 - nu[0]
    # add_line(
    #     axes[0],
    #     # 0.63,
    #     0.88,
    #     # 0.6,
    #     0.8,
    #     1,
    #     slope2,
    #     scale="log",
    #     # label=r"slope= $\beta - \nu$",
    #     label=None,
    #     xl=0.7,
    #     yl=0.5)
    plt.suptitle(
        r"$\eta=%g,\beta/\nu =%g$" % (eta, beta_over_nu), fontsize="xx-large")
    plt.show()
    plt.close()


def varied_alpha(eta, xi_m=100):
    phi_dict = get_phi_dict(eta)
    if eta == 0.1:
        alpha_arr = np.linspace(0.55, 0.9, 50)
    elif eta == 0.10:
        alpha_arr = np.linspace(0.5, 0.8, 50)
    nu_arr = np.zeros_like(alpha_arr)
    eps_c_arr = np.zeros_like(alpha_arr)
    nu_err = np.zeros_like(nu_arr)
    eps_c_err = np.zeros_like(nu_arr)
    n_arr = np.zeros(alpha_arr.size, int)
    for i, alpha in enumerate(alpha_arr):
        c0, res, eps_m, Lm, phi_m, xi_err = find_peak(eta, phi_dict, alpha,
                                                      False, False, None, True)
        mask = Lm > xi_m
        x = eps_m[mask]
        y = Lm[mask]
        popt, perr = fit_pow(x, y, xi_err=xi_err[mask])
        # popt, perr = fit_pow(x, y)
        eps_c_arr[i] = popt[0]
        nu_arr[i] = popt[2]
        eps_c_err[i] = perr[0]
        nu_err[i] = perr[2]
        n_arr[i] = x.size
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, constrained_layout=True)
    n_set = set(n_arr)
    for i in n_set:
        mask = n_arr == i
        label = r"$n=%d$" % i
        line1, = ax1.plot(alpha_arr[mask], eps_c_arr[mask], "o", label=label)
        line2, = ax2.plot(alpha_arr[mask], nu_arr[mask], "o", label=label)
        ax1.errorbar(
            alpha_arr[mask],
            eps_c_arr[mask],
            yerr=eps_c_err[mask],
            fmt="none",
            c=line1.get_color())
        ax2.errorbar(
            alpha_arr[mask],
            nu_arr[mask],
            yerr=nu_err[mask],
            fmt="none",
            c=line1.get_color())

    ax1.legend(fontsize="x-large")
    ax2.legend(fontsize="x-large")
    ax1.set_ylabel(r"$\epsilon_c$", fontsize="xx-large")
    ax2.set_ylabel(r"$\nu$", fontsize="xx-large")
    ax2.set_xlabel(r"$\alpha$", fontsize="xx-large")
    plt.suptitle(
        r"$\eta=%g,\xi_{\rm min}=%g$" % (eta, xi_m), fontsize="xx-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    eta = 0.18
    disorder_t = "RF"
    phi_dict = get_phi_dict(eta, 0, disorder_t=disorder_t)
    # plot_three_panel(eta, 0.6, save_fig=False, save_data=False)
    # plot_slope_vs_L(phi_dict, eps_max=0.035, eta=eta)
    collapse3(eta, 0.02, disorder_t)
    # collapse_phi_L(eta, phi_dict, eps_max=0.05, square_eps=False)
    # plot_collapse_phi_L(True)
