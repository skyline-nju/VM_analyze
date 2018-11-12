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
from numpy import polyfit
import os
import sys
sys.path.append("..")
try:
    from corr2d.add_line import add_line
except ImportError:
    print("error when import add_line")


def get_chi_dict(eta, is_dis=False, disorder_t="RT"):
    """
    Get dict of chi with key `L`. chi_dict[L] is a 2 * n array,
    eps_arr=chi_dict[eps][0], chi_arr = chi_dict[eps][1].

    Parameters
    --------
    eta: float
        Strength of noise.
    is_dis: bool
        If True, return xi_dis, else return xi.
    disorder_t: str
        Type of quenched disorder, 'RT' for random torque, 'RF' for
        random field.

    Returns:
    --------
    chi_dict: dict
        xi or xi_dis.
    """
    from create_dict import create_dict_from_xlsx
    if disorder_t == "RT":
        path = r"E:\data\random_torque\susceptibility\sample_average"
    elif disorder_t == "RF":
        path = r"E:\data\random_field\normalize_new\scaling\sample_average"
        # path = r"E:\data\random_field\normalize\scaling\sample_average"
    if not os.path.exists(path):
        path = path.replace("E:", "D:")

    infile = path + os.path.sep + r"eta=%g.xlsx" % eta
    if is_dis:
        chi_type = "chi_dis"
    else:
        chi_type = "chi"
    chi_dict = {}
    if disorder_t == "RT":
        if eta == 0.18:
            eps_min = 0.0455
            # eps_min = 0.01
            L_min = 46
            chi_dict = create_dict_from_xlsx(
                infile, chi_type, "L", eps_min, L_min)
            del_keys = []
            for L in chi_dict.keys():
                if L not in [46, 64, 90, 128, 180, 256, 362, 512, 724, 1024]:
                    del_keys.append(L)
                # if L == 256:
                #     chi_dict[L] = chi_dict[L][:, 1:-2]
                # if L == 724 or L == 512 or L == 362:
                #     chi_dict[L] = chi_dict[L][:, 1:-1]
            for L in del_keys:
                del chi_dict[L]
        elif eta == 0.10:
            eps_min = 0.045
            # eps_min = 0.01
            L_min = 45
            chi_dict = create_dict_from_xlsx(
                infile, chi_type, "L", eps_min, L_min)
            for L in chi_dict:
                if L > 90:
                    chi_dict[L] = chi_dict[L][:, :-2]
                    if L == 512:
                        chi_dict[L] = chi_dict[L][:, :-2]
            del chi_dict[54]
            del chi_dict[108]
        elif eta == 0.05:
            eps_min = 0.03
            L_min = 45
            chi_dict = create_dict_from_xlsx(
                infile, chi_type, "L", eps_min, L_min)
            # del chi_dict[180]
            # del chi_dict[256]
            # del chi_dict[362]
            del chi_dict[512]
    elif disorder_t == "RF":
        if eta == 0.18:
            print(infile)
            eps_min = 0.125
            # eps_min = 0.12
            L_min = 60
            chi_dict = create_dict_from_xlsx(
                infile, chi_type, "L", eps_min, L_min)
        del chi_dict[362]
        del chi_dict[512]
        # del chi_dict[724]
    return chi_dict


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


def find_peak(eta, chi_dict, ax=None, mode="con", disorder_t="RT"):
    from fit import find_peak_polyfit
    L_arr = np.array([i for i in sorted(chi_dict.keys())])
    eps_p = np.zeros(L_arr.size)
    chi_p = np.zeros_like(eps_p)
    eps_err = np.zeros_like(eps_p)
    chi_err = np.zeros_like(eps_p)
    for L in sorted(chi_dict.keys()):
        eps_arr, chi_arr = chi_dict[L]
        i = np.argwhere(L_arr == L)
        eps_p[i], chi_p[i], eps_err[i], chi_err[i], c = find_peak_polyfit(
            eps_arr, chi_arr, yscale="log")
        if ax is not None:
            x = np.linspace(eps_arr.min(), eps_arr.max(), 1000)
            y = np.exp(np.polyval(c, x))
            line, = ax.plot(eps_arr, chi_arr, "o", label=r"$%d$" % L)
            if ax is not None:
                ax.plot(x, y, color=line.get_color())
    if ax is not None:
        ax.plot(eps_p, chi_p, "ks--", fillstyle='none')
        ax.errorbar(eps_p, chi_p, yerr=chi_err, xerr=eps_err, fmt='none')
        ax.set_yscale("log")
        ax.legend(loc="upper right", title=r"$L=$")
        if disorder_t == "RT":
            ax.set_xlim(xmax=0.09)
        ax.set_xlabel(r"$\epsilon$", fontsize="xx-large")
        if mode == "con":
            yl = r"$L^2\left [\langle m^2\rangle-\langle m\rangle^2\right ]$"
        elif mode == "dis":
            yl = r"$L^2\left([\langle m\rangle^2]-[\langle m\rangle]^2\right)$"
        else:
            yl = r"$L^2\left([\langle m^2\rangle]-[\langle m\rangle]^2\right)$"
        ax.set_ylabel(yl, fontsize="xx-large")
    return eps_p, chi_p, eps_err, chi_err


def plot_chi_peak_vs_L(eta, L, chi_p, chi_err, slope, ax, mode="con"):
    ax.plot(L, chi_p, "o")
    x0, x1 = L[0] - 2, L[-1] + 50
    ax.errorbar(L, chi_p, yerr=chi_err, fmt="none")
    beg = 0
    c, V = polyfit(np.log10(L[beg:]), np.log10(chi_p[beg:]), deg=1, cov=True)
    print(c)
    print(V)
    x = np.linspace(L[beg], x1, 1000)
    y = 10**c[1] * x**c[0]
    ax.plot(x, y, "--", label=r"$\chi_{\rm p}\sim L^{%.4f}$" % c[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x0, x1)
    ax.legend(loc="upper left", title="linear fit", fontsize="x-large")
    label = r"$\chi_p \propto L^{%g}$" % slope
    add_line(ax, 0, 0.2, 1, slope, label=label, scale="log")
    ax.set_xlabel(r"$L$", fontsize="xx-large")
    ax.set_ylabel(
        r"${\rm susceptibility\ peak\ }\chi_{\rm p}$", fontsize="xx-large")


def plot_peak_loc_vs_L(eta, L, eps_p, eps_err, ax):
    """
        Plot susceptibility peak and its loaction against system size,
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
    ax.plot(L, eps_p, "o")
    ax.errorbar(L, eps_p, yerr=eps_err, fmt="none")
    from fit import plot_KT_fit, plot_pow_fit
    if eta == 0.18:
        eps_m = 0.05
    elif eta == 0.1:
        eps_m = 0.045
    else:
        eps_m = 0.02
    beg = 0
    x = eps_p[beg:]
    y = L[beg:]
    # x_err = eps_err[beg:]

    ax.axvspan(y[0] - 20, y[-1] + 100, alpha=0.2)
    plot_KT_fit(0.5, ax, x, y, reversed=True, eps_min=eps_m)
    plot_KT_fit(1.0, ax, x, y, reversed=True, eps_min=eps_m)
    plot_pow_fit(ax, x, y, reversed=True, eps_err=None)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L$", fontsize="xx-large")
    ax.set_ylabel(
        r"${\rm peak\ location\ }\epsilon_{\rm p}$", fontsize="xx-large")
    ax.legend(fontsize="large", title="fitting curve")


def read_suscept_peak(eta, head, tail):
    path = r"data\eta=%.2f" % eta
    with open(path + os.path.sep + "suscept_peak.dat") as f:
        lines = f.readlines()
        lines = lines[head:len(lines) - tail]
        L = np.zeros(len(lines))
        eps = np.zeros_like(L)
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            L[i] = float(s[0])
            eps[i] = float(s[1])
            print(L[i], eps[i])
    return L, eps


def plot_chi_con_and_dis(eta, disorder_t="RT"):
    chi_con_dict = get_chi_dict(eta, False, disorder_t=disorder_t)
    chi_dis_dict = get_chi_dict(eta, True, disorder_t=disorder_t)
    for L in sorted(chi_con_dict.keys()):
        eps_arr, chi_dis_arr = chi_dis_dict[L]
        eps_arr, chi_con_arr = chi_con_dict[L]
        line, = plt.plot(eps_arr, chi_con_arr, "o",
                         label="$%d$" % L)
        plt.plot(eps_arr, chi_dis_arr, "--s",
                 fillstyle="none", c=line.get_color())
    plt.yscale("log")
    plt.legend(title=r"$L=$")
    plt.xlabel(r"$\epsilon$", fontsize="xx-large")
    plt.ylabel(r"$\chi_\mathrm{con}(\chi_\mathrm{dis})$", fontsize="xx-large")
    plt.title(r"random torque with $\eta=0.18, \rho_0=1$", fontsize="xx-large")
    plt.show()
    plt.close()


def plot_3_panels(eta, save_fig=False, mode="con", disorder_t="RT"):
    """
    Plot three panels:
    1) sample-averaged chi vs. eps with increasing L and eta = eta.
    2) susceptibility peak vs. L.
    3) location of susceptibility peak vs. L.

    """
    if mode == "con":
        chi_dict = get_chi_dict(eta, False, disorder_t)
    elif mode == "dis":
        chi_dict = get_chi_dict(eta, True, disorder_t)
    elif mode == "mix":
        chi_dict = {}
        chi_con = get_chi_dict(eta, False, disorder_t)
        chi_dis = get_chi_dict(eta, True, disorder_t)
        for L in chi_con:
            eps_arr, chi_arr1 = chi_dis[L]
            eps_arr, chi_arr2 = chi_con[L]
            chi_dict[L] = [eps_arr, chi_arr1 + chi_arr2]
    if eta == 0.0 or disorder_t == "RF":
        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
    else:
        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)
    L_arr = np.array([i for i in sorted(chi_dict.keys())])
    eps_p, chi_p, eps_err, chi_err = find_peak(
        eta, chi_dict, axes[0], mode, disorder_t)
    axes[0].set_title("(a)", fontsize="xx-large")
    if mode == "dis":
        if eta == 0.18:
            # slope = 2.22
            slope = 2.46
        elif eta == 0.1:
            slope = 2.1
        elif eta == 0.05:
            slope = 2.52
    elif mode == "con":
        if eta == 0.18:
            slope = 1.75
            # slope = 1.48
        elif eta == 0.1:
            slope = 1.8
        elif eta == 0.05:
            slope = 1.65
    else:
        if eta == 0.05:
            slope = 1.7
        else:
            slope = 1.96
    plot_chi_peak_vs_L(eta, L_arr, chi_p, chi_err, slope, axes[1], mode)
    axes[1].set_title("(b)", fontsize="xx-large")
    if eta != 0.0 and disorder_t == "RT":
        plot_peak_loc_vs_L(eta, L_arr, eps_p, eps_err, axes[2])
        axes[2].set_title("(c)", fontsize="xx-large")
        ax_in = fig.add_axes([0.55, 0.2, 0.1, 0.3])
    else:
        ax_in = fig.add_axes([0.8, 0.18, 0.18, 0.32])

    ax_in.loglog(L_arr, chi_p / L_arr**slope, "o")
    ylabel = r"$\chi_p / L^{%g}$" % slope
    ax_in.text(0.05, 0.8, ylabel, transform=ax_in.transAxes, fontsize="large")
    ax_in.text(0.85, 0.05, r"$L$", transform=ax_in.transAxes, fontsize="large")
    title = r"$\eta=%g,$" % eta
    if mode == "con":
        title += " connected susceptibility"
    elif mode == "dis":
        title += " disconnected susceptibility"
    else:
        title += " total susceptibility"
    plt.suptitle(title, fontsize="xx-large")
    if save_fig:
        plt.savefig(r"data\suscept_peak_eta=%g.eps" % eta)
    else:
        plt.show()


def collapse_suscept(eta):
    chi_dict = get_chi_dict(eta, False)
    if eta == 0.18:
        gamma_over_nu = 1.7489
        eps_c1 = 0.0448
        nu1 = 1.873
        eps_c2 = 0.0345
        nu2 = 1
        A2 = 8.005
        eps_c3 = 0.0397
        nu3 = 0.5
        A3 = 1.258
    elif eta == 0.1:
        gamma_over_nu = 1.798
        eps_c1 = 0.0397
        nu1 = 1.988
        nu2 = 1
        eps_c2 = 0.0292
        A2 = 5.844
        nu3 = 0.5
        eps_c3 = 0.0345
        A3 = 0.813
    else:
        print("eta should be one of 0.10 or 0.18!")
        sys.exit()

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4.5),
        constrained_layout=True,
        sharey=True)
    for L in sorted(chi_dict.keys()):
        if L < 128:
            continue
        eps_arr, chi_arr = chi_dict[L]
        y = chi_arr * L**(-gamma_over_nu)
        x1 = np.absolute(eps_arr - eps_c1) * L**(1 / nu1)
        x2 = (np.log(L) - np.log(A2))**(
            1 / nu2) * np.absolute(eps_arr - eps_c2)
        x3 = (np.log(L) - np.log(A3))**(
            1 / nu3) * np.absolute(eps_arr - eps_c3)
        ax1.plot(x1, y, "o", label="%d" % L, fillstyle="none")
        ax2.plot(x2, y, "o", label="%d" % L, fillstyle="none")
        ax3.plot(x3, y, "o", label="%d" % L, fillstyle="none")
    ax1.legend(title=r"$L=$", fontsize="large")
    ax2.legend(title=r"$L=$", fontsize="large")
    ax3.legend(title=r"$L=$", fontsize="large")
    ylabel = r"$\chi L ^ {-\gamma / \nu}$"
    xlabel = r"$|\epsilon-\epsilon_c| L ^ {1/\nu}$"
    xlabel2 = r"$|\epsilon-\epsilon_c|\left[\ln L - \ln A_\chi\right]^{1/\nu}$"
    ax1.set_ylabel(ylabel, fontsize="x-large")
    ax1.set_xlabel(xlabel, fontsize="x-large")
    ax2.set_xlabel(xlabel2, fontsize="x-large")
    ax3.set_xlabel(xlabel2, fontsize="x-large")
    ax1.set_title(r"(a) algebraic scaling")
    ax2.set_title(r"(b) KT scaling")
    ax3.set_title(r"(c) KT scaling")
    # ax1.set_yscale("log")
    pat = r"$\nu=%g$" + "\n" + r"$\epsilon_c=%g$"
    ax1.text(
        0.65,
        0.8,
        pat % (nu1, eps_c1),
        transform=ax1.transAxes,
        fontsize="x-large")
    pat += "\n" + r"$A_\chi=%g$"
    ax2.text(
        0.65,
        0.75,
        pat % (nu2, eps_c2, A2),
        transform=ax2.transAxes,
        fontsize="x-large")
    ax3.text(
        0.65,
        0.75,
        pat % (nu3, eps_c3, A3),
        transform=ax3.transAxes,
        fontsize="x-large")
    plt.suptitle(
        r"$\eta=%g,\gamma / \nu =%g$" % (eta, gamma_over_nu),
        fontsize="xx-large")
    plt.show()
    plt.close()


def plot_chi_mix(eta):
    chi_con = get_chi_dict(eta, False)
    chi_dis = get_chi_dict(eta, True)
    chi_mix = {}
    for L in chi_con:
        eps_arr, chi_arr1 = chi_dis[L]
        eps_arr, chi_arr2 = chi_con[L]
        chi_mix[L] = [eps_arr, chi_arr1 + chi_arr2]
    for L in chi_mix:
        eps_arr, chi_arr = chi_mix[L]
        plt.plot(eps_arr, chi_arr, "o", label=r"$L=%d$" % L)
    plt.legend()
    plt.yscale("log")
    plt.show()
    plt.close()


def plot_chi_mix_dis(disorder_t="RT"):
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(9, 4), constrained_layout=True)
    c = []
    if disorder_t == "RT":
        eta_list = [0.05, 0.1, 0.18]
    else:
        eta_list = [0.18]
    for i, eta in enumerate(eta_list):
        chi_con_dict = get_chi_dict(eta, False, disorder_t=disorder_t)
        chi_dis_dict = get_chi_dict(eta, True, disorder_t=disorder_t)
        chi_mix_dict = {}
        for L in chi_con_dict:
            eps, chi1 = chi_dis_dict[L]
            eps, chi2 = chi_con_dict[L]
            chi_mix_dict[L] = [eps, chi1 + chi2]
        L_arr = np.array([L for L in sorted(chi_mix_dict.keys())])
        eps_p_dis, chi_p_dis, eps_err_dis, chi_err_dis = find_peak(
            eta, chi_dis_dict)
        eps_p_mix, chi_p_mix, eps_err_mix, chi_err_mix = find_peak(
            eta, chi_mix_dict)
        eps_p_con, chi_p_con, eps_err_con, chi_err_con = find_peak(
            eta, chi_con_dict)
        ax1.plot(L_arr, chi_p_mix / chi_p_dis, "o", label=r"$\eta=%g$" % eta)
        line, = ax2.plot(
            L_arr, 1 - chi_p_dis / chi_p_mix, "o", label=r"$\eta=%g$" % eta)
        c.append(line.get_color())

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    if disorder_t == "RT":
        lb = r"$L^{-0.07}$"
        add_line(ax2, 0.4, 0.95, 1, -0.07, scale="log",
                c=c[0], label=lb, xl=0.7, yl=0.9)
        lb = r"$L^{-0.085}$"
        add_line(ax2, 0, 0.85, 1, -0.085, scale="log",
                c=c[1], label=lb, xl=0.5, yl=0.7)
        lb = r"$L^{-0.16}$"
        add_line(
            ax2, 0, 0.83, 1, -0.16, scale="log", c=c[2], label=lb, xl=0.3, yl=0.45)
    else:
        # ax2.set_ylim(0.1)
        # ax2.set_xlim(xmax=3000)
        lb = r"$L^{-0.5}$"
        add_line(ax2, 0., 1, 1, -0.5, scale="log", c=c[0], label=lb)
    ax1.set_xlabel(r"$L$", fontsize="x-large")
    ax1.set_ylabel(r"$\chi_{\rm tot}/\chi_{\rm dis}$", fontsize="x-large")
    ax2.set_xlabel(r"$L$", fontsize="x-large")
    ax2.set_ylabel(
        r"$1 - \frac{\chi_{\rm dis}} {\chi_{\rm tot}}$", fontsize="xx-large")
    ax1.legend(fontsize="x-large")
    ax2.legend(fontsize="x-large", loc="lower left")
    plt.show()
    plt.close()


def xlsx_to_txt(eta):
    chi_con = get_chi_dict(eta, False)
    chi_dis = get_chi_dict(eta, True)
    out_file_pat = r"eta=%g_L=%d.dat"
    for L in chi_con:
        out_file = out_file_pat % (eta, L)
        print(out_file)
        with open(out_file, "w") as f:
            eps_arr, chi_con_arr = chi_con[L]
            eps_arr, chi_dis_arr = chi_dis[L]
            f.writelines("%f\t%f\t%f\n" % (
                eps_arr[i], chi_con_arr[i], chi_dis_arr[i]
            ) for i in range(eps_arr.size))


def fit_w_fixed_nu(mode="con", first=3, last=None):
    """ Do fitting with fixed nu. """
    from fit import fit_pow2, plot_pow_fit
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for eta in [0.1, 0.18]:
        if mode == "con":
            chi_dict = get_chi_dict(eta, False)
        elif mode == "dis":
            chi_dict = get_chi_dict(eta, True)
        elif mode == "mix":
            chi_dict = {}
            chi_con = get_chi_dict(eta, False)
            chi_dis = get_chi_dict(eta, True)
            for L in chi_con:
                eps_arr, chi_arr1 = chi_dis[L]
                eps_arr, chi_arr2 = chi_con[L]
                chi_dict[L] = [eps_arr, chi_arr1 + chi_arr2]
        L_arr = np.array([i for i in sorted(chi_dict.keys())])
        eps_p, chi_p, eps_err, chi_err = find_peak(eta, chi_dict)
        x = L_arr[first:last]
        y = eps_p[first:last]
        y_err = eps_err[first:last]
        nu_arr = np.linspace(1.5, 3.5, 50)
        eps_arr = np.zeros_like(nu_arr)
        err_arr = np.zeros_like(nu_arr)
        ax1.plot(x, y, "o", label=r"$\eta=%g$" % eta)
        for i, nu in enumerate(nu_arr):
            popt, perr = fit_pow2(x, y, y_err, nu0=nu)
            eps_arr[i] = popt[0]
            err_arr[i] = perr[0]
        line, = ax2.plot(nu_arr, eps_arr, "o", label=r"$\eta=%g$" % eta, ms=2)
        ax2.errorbar(nu_arr, eps_arr, err_arr, c=line.get_color())
        x_fit, y_fit = plot_pow_fit(None, y, x, reversed=True, eps_err=y_err,
                                    nu0=2.84)
        ax1.plot(x_fit, y_fit, "--")
        x_fit, y_fit = plot_pow_fit(None, y, x, reversed=True, eps_err=y_err)
        ax1.plot(x_fit, y_fit)
    ax1.set_xscale("log")
    ax2.set_xlabel(r"$\nu$", fontsize="x-large")
    ax2.set_ylabel(r"$\epsilon_c$", fontsize="x-large")
    ax2.set_title("algebraic scaling", fontsize="x-large")
    plt.tight_layout()
    ax2.axhline(0.04)
    ax1.legend(fontsize="x-large")
    ax2.legend(fontsize="x-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    eta = 0.18
    plot_3_panels(eta, save_fig=False, mode="dis", disorder_t="RF")
    # fit_w_fixed_nu()
    # collapse_suscept(eta)
    # plot_chi_mix(eta)

    # plot_chi_mix_dis("RF")
    # xlsx_to_txt(eta)
    # plot_chi_con_and_dis(eta, disorder_t="RF")
