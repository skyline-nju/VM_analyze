"""
Estimate the correlation length beyond which the order parameter begin to decay
with a power law.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.polynomial.polynomial import polyfit
from add_line import add_line
from suscept_peak import find_peak_by_polyfit
from fit import plot_KT_fit, plot_pow_fit, fit_exp


def get_phi_dict(eta):
    """
    Get dict of phi with key `eps`. phi_dict[eps] is a 2 * n array,
    L_arr=phi_dict[eps][0], phi_arr = phi_dict[eps][1].
    """
    if eta == 0.18:
        from create_dict import create_dict_from_txt
        path = r"data\eta=%.2f" % eta
        eps_min = 0.0535
        L_min = None
        phi_dict = create_dict_from_txt(path, "phi", "eps", eps_min, L_min,
                                        "dict-arr", 5)
    else:
        from create_dict import create_dict_from_xlsx
        path = r"E:\data\random_torque\susceptibility"
        infile = path + os.path.sep + r"eta=%g.xlsx" % eta
        eps_min = 0.048
        L_min = None
        phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min, L_min,
                                         "dict-arr", 5)
        # del phi_dict[0.065]
        del phi_dict[0.054]
    return phi_dict


def plot_phi_vs_L(phi_dict, ax=None, eta=None, Lc=None, phi_c=None):
    """ PLot phi against L with varied epsilon in log-log scales."""
    if ax is None:
        ax = plt.gca()
        flag_show = True
    else:
        flag_show = False
    color = plt.cm.gist_rainbow(np.linspace(0, 1, len(phi_dict)))
    for i, eps in enumerate(sorted(phi_dict.keys())):
        L = phi_dict[eps][0]
        phi = phi_dict[eps][1]
        ax.plot(L, phi, "-o", label="%.4f" % eps, color=color[i])
    if Lc is not None and phi_c is not None:
        ax.plot(Lc, phi_c, "--ks", fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$", fontsize="x-large")
    ax.set_ylabel(r"$\phi$", fontsize="x-large")
    ax.legend(title=r"$\epsilon=$", loc="lower left")
    add_line(ax, 0.25, 0.65, 0.55, -1, scale="log")
    if flag_show:
        plt.tight_layout()
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
    color = plt.cm.gist_rainbow(np.linspace(0, 1, xm.size))
    for i, key in enumerate(sorted(phi_dict.keys())):
        L = phi_dict[key][0]
        phi = phi_dict[key][1]
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
    if save_data:
        with open(r"data\eta=%.2f\polar_order.dat" % eta, "w") as f:
            for i in range(eps_m.size):
                f.write("%.4f\t%.8f\n" % (eps_m[i], xm[i]))
    if show:
        ax.plot(xm, ym, "ks", fillstyle="none")
        ax.plot(10**x, 10**y, "k--", label=r"$L^{%.3f}$" % (c[1]))
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
        return c[1], stats[0][0], eps_m, xm, phi_m


def find_peak2(phi_dict, alpha, ret_fit=False):
    """ Find the peak of phi * L ** alpha vs L in log-log scales."""
    Lm = np.zeros(len(phi_dict))
    ym = np.zeros_like(Lm)
    eps_m = np.zeros_like(Lm)
    size_fit = 10000
    if ret_fit:
        x_fit = np.zeros((Lm.size, size_fit))
        y_fit = np.zeros_like(x_fit)
    for i, eps in enumerate(sorted(phi_dict.keys())):
        eps_m[i] = eps
        L, phi = phi_dict[eps]
        y = phi * L**alpha
        if ret_fit:
            Lm[i], ym[i], x_fit[i], y_fit[i] = find_peak_by_polyfit(
                L, y, 5, "log", "log", True, size_fit)
        else:
            Lm[i], ym[i] = find_peak_by_polyfit(L, y, 5, "log", "log")
    if ret_fit:
        return Lm, ym, eps_m, x_fit, y_fit
    else:
        return Lm, ym, eps_m


def plot_three_panel(eta, phi_dict, alpha, save_fig=False, save_data=False):
    """ Plot phi vs. L, L^alpha * phi vs. L and correlation length vs. eps."""
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    c0, res, eps_m, Lm, phi_m = find_peak(eta, phi_dict, alpha, True,
                                          save_data, ax2, True)
    plot_phi_vs_L(phi_dict, ax1, eta, Lm, phi_m)
    ax3.plot(eps_m, Lm, "o")

    plot_KT_fit(0.5, ax3, eps_m, Lm)
    plot_KT_fit(1.0, ax3, eps_m, Lm)
    plot_pow_fit(ax3, eps_m, Lm)

    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\epsilon$", fontsize="x-large")
    ax3.set_ylabel(r"$\xi$", fontsize="x-large")
    ax3.legend(fontsize="large", title="fitting curve")
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    ax3.set_title("(c)")
    fig.tight_layout()
    if save_fig:
        plt.savefig(r"data\eta=%.2f\polar_order.eps" % eta)
    else:
        plt.show()
    plt.close()


def varied_alpha(nus, phi_dict):
    alpha = np.linspace(0.4, 0.6, 100)
    squared_res = np.zeros_like(alpha)
    for nu in nus:
        for i in range(alpha.size):
            Lm, ym, eps_m = find_peak2(phi_dict, alpha[i])
            popt, perr, squared_res[i] = fit_exp(
                eps_m, Lm, beta=nu, ret_res=True)
        plt.plot(alpha, squared_res, label="%g" % nu)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Squared residuals for KT-like fitting")
    plt.legend(title=r"$\nu=$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    eta = 0.1
    phi_dict = get_phi_dict(eta)
    # eta = 0.18
    # os.chdir("data/eta=%.2f" % eta)
    # varied_alpha([0.4, 0.6, 0.8, 1.0, 1.2])
    # find_peak(0.5, save_data=True, eps_min=0.05)
    # plot_phi_vs_L()
    # plot_L_vs_eps_c([0.35, 0.4, 0.45, 0.5])
    # plot_phi_vs_L(phi_dict, None, eta, None, None)
    plot_three_panel(eta, phi_dict, 0.5, save_fig=False, save_data=True)
