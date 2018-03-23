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
except:
    print("error when import add_line")


def get_phi_dict(eta):
    """
    Get dict of phi with key `eps`. phi_dict[eps] is a 2 * n array,
    L_arr=phi_dict[eps][0], phi_arr = phi_dict[eps][1].
    """
    if eta == 0.18:
        from create_dict import create_dict_from_txt
        path = r"data\eta=%.2f" % eta
        # eps_min = 0.01
        eps_min = 0.0535
        L_min = None
        phi_dict = create_dict_from_txt(path, "phi", "eps", eps_min, L_min,
                                        "dict-arr", 5)
    else:
        from create_dict import create_dict_from_xlsx
        path = r"E:\data\random_torque\susceptibility"
        infile = path + os.path.sep + r"eta=%g.xlsx" % eta
        # eps_min = 0.01
        eps_min = 0.048
        L_min = None
        phi_dict = create_dict_from_xlsx(infile, "phi", "eps", eps_min, L_min,
                                         "dict-arr", 5)
        del phi_dict[0.053]
        del phi_dict[0.055]
        # del phi_dict[0.048]
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


def plot_three_panel(eta, phi_dict, alpha, save_fig=False, save_data=False):
    """ Plot phi vs. L, L^alpha * phi vs. L and correlation length vs. eps."""
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
        x = eps_m[:-4]
        y = Lm[:-4]
    elif eta == 0.1:
        eps_min = 0.047
        eps_max = 0.076
        x = eps_m[:-7]
        y = Lm[:-7]

    ax3.axhspan(y[-1] - 10, y[0] + 100, alpha=0.2)
    plot_KT_fit(0.5, ax3, x, y, eps_min=eps_min, eps_max=eps_max)
    plot_KT_fit(1.0, ax3, x, y, eps_min=eps_min, eps_max=eps_max)
    plot_pow_fit(ax3, x, y, eps_min=eps_min, eps_max=eps_max)
    print(Lm[:-5])

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


def collapse(eta, phi_dict):
    if eta == 0.18:
        beta_over_nu = 0.05
        eps_c1 = 0.0443
        nu1 = 1.829
        eps_c2 = 0.0306
        nu2 = 1
        A2 = 5.246
        eps_c3 = 0.0375
        nu3 = 0.5
        A3 = 0.851
        eps_min = 0.07
    elif eta == 0.1:
        beta_over_nu = 0.05
        eps_c1 = 0.0403
        nu1 = 2.048
        nu2 = 1
        eps_c2 = 0.0307
        A2 = 3.695
        nu3 = 0.5
        eps_c3 = 0.0355
        A3 = 0.48
        eps_min = 0.057

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4.5),
        constrained_layout=True,
        sharey=True)
    for eps in sorted(phi_dict.keys()):
        if eps >= eps_min:
            continue
        L_arr, phi_arr = phi_dict[eps]
        y = phi_arr * L_arr**(beta_over_nu)
        x1 = np.absolute(eps - eps_c1) * L_arr**(1 / nu1)
        x2 = (np.log(L_arr) - np.log(A2))**(
            1 / nu2) * np.absolute(eps - eps_c2)
        x3 = (np.log(L_arr) - np.log(A3))**(
            1 / nu3) * np.absolute(eps - eps_c3)
        if eps < eps_c1:
            ax1.plot(x1, y, "s", label="%.3f" % eps, fillstyle="none")
        else:
            ax1.plot(x1, y, "o", label="%.3f" % eps, fillstyle="none")
        if eps < eps_c2:
            ax2.plot(x2, y, "s", label="%.3f" % eps, fillstyle="none")
        else:
            ax2.plot(x2, y, "o", label="%.3f" % eps, fillstyle="none")
        if eps < eps_c3:
            ax3.plot(x3, y, "s", label="%.3f" % eps, fillstyle="none")
        else:
            ax3.plot(x3, y, "o", label="%.3f" % eps, fillstyle="none")
    ax1.legend(
        title=r"$\epsilon=$",
        loc="lower right",
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.05)
    ax2.legend(
        title=r"$\epsilon=$",
        loc="lower left",
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.05)
    ax3.legend(
        title=r"$\epsilon=$",
        loc="center right",
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.05)
    ylabel = r"$\phi L ^ {\beta / \nu}$"
    xlabel = r"$|\epsilon-\epsilon_c| L ^ {1/\nu}$"
    xlabel2 = r"$|\epsilon-\epsilon_c|\left[\ln L - \ln A_\xi\right]^{1/\nu}$"
    ax1.set_ylabel(ylabel, fontsize="x-large")
    ax1.set_xlabel(xlabel, fontsize="x-large")
    ax2.set_xlabel(xlabel2, fontsize="x-large")
    ax3.set_xlabel(xlabel2, fontsize="x-large")
    ax1.set_title(r"(a) algebraic scaling")
    ax2.set_title(r"(b) KT scaling")
    ax3.set_title(r"(c) KT scaling")
    if eta == 0.18:
        ax1.set_xlim(xmax=1.5)
        pass
    pat = r"$\nu=%g$" + "\n" + r"$\epsilon_c=%g$"
    ax1.text(
        0.65,
        0.7,
        pat % (nu1, eps_c1),
        transform=ax1.transAxes,
        fontsize="x-large")
    pat += "\n" + r"$A_\xi=%g$"
    ax2.text(
        0.65,
        0.6,
        pat % (nu2, eps_c2, A2),
        transform=ax2.transAxes,
        fontsize="x-large")
    ax3.text(
        0.1,
        0.1,
        pat % (nu3, eps_c3, A3),
        transform=ax3.transAxes,
        fontsize="x-large")
    plt.suptitle(
        r"$\eta=%g,\beta/\nu =%g$" % (eta, beta_over_nu), fontsize="xx-large")
    plt.show()
    plt.close()


def varied_alpha(eta, phi_dict, xi_m=100):
    if eta == 0.18:
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
    phi_dict = get_phi_dict(eta)
    # plot_phi_vs_L()
    # plot_L_vs_eps_c([0.35, 0.4, 0.45, 0.5])
    # plot_phi_vs_L(phi_dict, None, eta, None, None)
    # xi_vs_eps_varied_alpha(eta, phi_dict)
    # changing_alpha(phi_dict)
    # powfit_varied_alpha(eta, phi_dict, h1=3, t1=0, h2=[0,3,0], t2=[0,0,3])
    plot_three_panel(eta, phi_dict, 0.85, save_fig=False, save_data=True)
    # collapse(eta, phi_dict)
    # varied_alpha(eta, phi_dict)
