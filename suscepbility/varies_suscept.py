import matplotlib.pyplot as plt
import os
import numpy as np
from create_dict import create_dict_from_xlsx
from cal_phi_chi import get_chi_M
import sys
sys.path.append("..")
try:
    from corr2d.add_line import add_line
except ImportError:
    print("error when import add_line")


def get_data(eta, disorder_t="RT"):
    """
    Get dict of phi with key `eps`. phi_dict[eps] is a 2 * n array,
    L_arr=phi_dict[eps][0], phi_arr = phi_dict[eps][1].
    """
    path = r"D:\data\VM2d\figure2"

    infile = path + os.path.sep + r"%s_eta=%g.xlsx" % (disorder_t, eta)

    Lmin = None
    phi_dict = create_dict_from_xlsx(infile, "phi", "eps", None, Lmin,
                                     "dict-arr", 0)
    chi_con_dict = create_dict_from_xlsx(infile, "chi", "eps", None, Lmin,
                                         "dict-arr", 0)
    chi_dis_dict = create_dict_from_xlsx(infile, "chi_dis", "eps", None, Lmin,
                                         "dict-arr", 0)

    return phi_dict, chi_con_dict, chi_dis_dict


def get_slope(L, phi):
    slope, L_mid = np.zeros((2, L.size - 1))
    for i in range(1, L.size):
        slope[i - 1] = np.log(phi[i - 1] / phi[i]) / np.log(L[i] / L[i - 1])
        L_mid[i - 1] = np.sqrt(L[i] * L[i - 1])
    return L_mid, slope


def plot_1():
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(6.4, 6.4), sharex="col")
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    ax3.set_yscale("log")
    ax4.set_yscale("log")
    # eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
    eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04, 0.045]
    # eps_arr = [0.05, 0.06, 0.07, 0.08]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(0.18, "RT")
    lines = []
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        R = chi_dis / L**2 / phi**2
        mask = L > 0
        line, = ax1.plot(L[mask], R[mask], "-o", label="%g" % eps)
        L_mid, slope = get_slope(L[mask], phi[mask])
        ax3.plot(
            L_mid,
            slope,
            "s",
            fillstyle="none",
            c=line.get_c(),
            label="%g" % eps)
        lines.append(line)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(r"(a) RT: $\eta=0.18$")
    phi_dict, chi_con_dict, chi_dis_dict = get_data(0.10, "RT")
    eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04]

    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        R = chi_dis / L**2 / phi**2
        L_mid, slope = get_slope(L, phi)

        line, = ax2.plot(L, R, "-o")
        ax4.plot(L_mid, slope, "s", fillstyle="none", c=line.get_c())
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(r"(b) RT: $\eta=0.10$")
    plt.tight_layout()

    ax1.text(0.02, 0.93, r"$R_M$", fontsize="x-large", transform=ax1.transAxes)
    ax2.text(0.02, 0.93, r"$R_M$", fontsize="x-large", transform=ax2.transAxes)
    ax3.text(
        0.02, 0.93, r"$\sigma$", fontsize="x-large", transform=ax3.transAxes)
    ax4.text(
        0.02, 0.93, r"$\sigma$", fontsize="x-large", transform=ax4.transAxes)
    ax1.text(0.95, 0.05, r"$L$", fontsize="x-large", transform=ax1.transAxes)
    ax2.text(0.95, 0.05, r"$L$", fontsize="x-large", transform=ax2.transAxes)
    ax3.text(0.95, 0.05, r"$L$", fontsize="x-large", transform=ax3.transAxes)
    ax4.text(0.95, 0.05, r"$L$", fontsize="x-large", transform=ax4.transAxes)

    # add_line(ax1, 0.2, 1, 1, -2, scale="log", label=r"$L^{-2}$", xl=0.65)
    # add_line(ax2, 0.2, 1, 1, -2, scale="log", label=r"$L^{-2}$", xl=0.65)
    ax3.legend(
        handles=lines, loc=(0.01, 0.01), frameon=False, labelspacing=0.1)

    plt.show()
    plt.close()


def plot_2(eta, disorder_t="RT"):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 8), sharex=True)
    axes[0][0].tick_params(direction='in', which="both")
    axes[0][1].tick_params(direction='in', which="both")
    axes[1][0].tick_params(direction='in', which="both")
    axes[1][1].tick_params(direction='in', which="both")

    # eps_arr = [0.05, 0.06, 0.07, 0.08]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04, 0.045]
            # eps_arr = [0.05, 0.06, 0.07, 0.08]
        elif eta == 0.1:
            eps_arr = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
            # eps_arr = [0.05, 0.06, 0.07, 0.075]
        elif eta == 0.05:
            # eps_arr = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035]
            eps_arr = [0.04, 0.045, 0.05, 0.055]
    elif disorder_t == "RF":
        if eta == 0.18:
            # eps_arr = [0.06, 0.07, 0.08, 0.1, 0.12]
            eps_arr = [0.13, 0.14, 0.15, 0.16]

    lines = []
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        chi_dis /= (L**2 * phi**2)
        chi_con /= (L**2 * phi**2)
        chi_tot = chi_dis + chi_con
        mask = L < 2048
        line, = axes[0][0].plot(L[mask], chi_dis[mask], "-o", label="%g" % eps)
        c = line.get_c()
        L_mid, slope = get_slope(L[mask], phi[mask])
        axes[0][1].plot(L[mask], chi_con[mask], "-o", label="%g" % eps, c=c)
        axes[1][0].plot(L[mask], chi_tot[mask], "-o", label="%g" % eps, c=c)
        axes[1][1].plot(
            L_mid, slope, "s", fillstyle="none", c=c, label="%g" % eps)
        lines.append(line)
    if disorder_t == "RT":
        axes[0][1].legend(frameon=False, labelspacing=0.05)
    elif disorder_t == "RF":
        axes[0][0].legend(frameon=False, labelspacing=0.05)
    axes[0][0].set_xscale("log")
    axes[0][0].set_yscale("log")
    axes[0][1].set_xscale("log")
    axes[0][1].set_yscale("log")
    axes[1][0].set_xscale("log")
    axes[1][0].set_yscale("log")
    axes[1][1].set_xscale("log")
    axes[1][1].set_yscale("log")
    # axes[0][0].set_xlabel(r"$L$", fontsize="x-large")
    # axes[0][1].set_xlabel(r"$L$", fontsize="x-large")
    axes[1][0].set_xlabel(r"$L$", fontsize="x-large")
    axes[1][1].set_xlabel(r"$L$", fontsize="x-large")
    axes[0][0].set_ylabel(r"$R$", fontsize="x-large")
    axes[0][1].set_ylabel(r"$\chi_{\rm con}/L^2M^2$", fontsize="x-large")
    axes[1][0].set_ylabel(r"$\chi_{\rm tot}/L^2M^2$", fontsize="x-large")
    axes[1][1].set_ylabel(r"$\sigma$", fontsize="x-large")
    # ax1.set_title(r"(a) RT: $\eta=0.18$")
    plt.tight_layout(rect=[-0.015, -0.025, 1.02, 0.98])
    if disorder_t == "RT":
        title = r"RS: $\eta=%g$" % eta
    else:
        title = r"%s: $\eta=%g$" % (disorder_t, eta)
    plt.suptitle(title, y=0.998, fontsize="x-large")
    plt.show()
    plt.close()


def plot_3(eta, disorder_t="RT"):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 8), sharex=True)
    axes[0][0].tick_params(direction='in', which="both")
    axes[0][1].tick_params(direction='in', which="both")
    axes[1][0].tick_params(direction='in', which="both")
    axes[1][1].tick_params(direction='in', which="both")

    # eps_arr = [0.05, 0.06, 0.07, 0.08]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            # eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04, 0.045]
            eps_arr = [0.05, 0.06, 0.07, 0.08]
        elif eta == 0.1:
            # eps_arr = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
            eps_arr = [0.05, 0.06, 0.07, 0.075]
        elif eta == 0.05:
            # eps_arr = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035]
            eps_arr = [0.04, 0.045, 0.05, 0.055]
    elif disorder_t == "RF":
        if eta == 0.18:
            # eps_arr = [0.06, 0.07, 0.08, 0.1, 0.12]
            eps_arr = [0.13, 0.14, 0.15, 0.16]

    lines = []
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        chi_dis /= (L**2)
        chi_con /= (L**2)
        chi_tot = chi_dis + chi_con
        if disorder_t == "RT":
            mask = L < 1448
        else:
            mask = L < 2048
        line, = axes[0][0].plot(L[mask], chi_dis[mask], "-o", label="%g" % eps)
        c = line.get_c()
        L_mid, slope = get_slope(L[mask], phi[mask])
        axes[0][1].plot(L[mask], chi_con[mask], "-o", label="%g" % eps, c=c)
        axes[1][0].plot(L[mask], chi_tot[mask], "-o", label="%g" % eps, c=c)
        axes[1][1].plot(
            L_mid, slope, "s", fillstyle="none", c=c, label="%g" % eps)
        lines.append(line)
    if disorder_t == "RT":
        axes[0][1].legend(frameon=False, labelspacing=0.05)
    elif disorder_t == "RF":
        axes[0][0].legend(frameon=False, labelspacing=0.05)
    axes[0][0].set_xscale("log")
    axes[0][0].set_yscale("log")
    axes[0][1].set_xscale("log")
    axes[0][1].set_yscale("log")
    axes[1][0].set_xscale("log")
    axes[1][0].set_yscale("log")
    axes[1][1].set_xscale("log")
    axes[1][1].set_yscale("log")
    # axes[0][0].set_xlabel(r"$L$", fontsize="x-large")
    # axes[0][1].set_xlabel(r"$L$", fontsize="x-large")
    axes[1][0].set_xlabel(r"$L$", fontsize="x-large")
    axes[1][1].set_xlabel(r"$L$", fontsize="x-large")
    axes[0][0].set_ylabel(r"$\chi_{\rm dis}/L^2$", fontsize="x-large")
    axes[0][1].set_ylabel(r"$\chi_{\rm con}/L^2$", fontsize="x-large")
    axes[1][0].set_ylabel(r"$\chi_{\rm tot}/L^2$", fontsize="x-large")
    axes[1][1].set_ylabel(r"$\sigma$", fontsize="x-large")
    # ax1.set_title(r"(a) RT: $\eta=0.18$")
    plt.tight_layout(rect=[-0.015, -0.025, 1.02, 0.98])
    if disorder_t == "RT":
        title = r"RS: $\eta=%g$" % eta
    else:
        title = r"%s: $\eta=%g$" % (disorder_t, eta)
    plt.suptitle(title, y=0.998, fontsize="x-large")
    plt.show()
    plt.close()


def plot_4(eta, disorder_t="RT", rescaled=True):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6), sharex=True)
    axes[0].tick_params(direction='in', which="both")
    axes[1].tick_params(direction='in', which="both")
    axes[2].tick_params(direction='in', which="both")
    axes[3].tick_params(direction='in', which="both")

    # eps_arr = [0.05, 0.06, 0.07, 0.08]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            eps_arr = [
                0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.035, 0.037, 0.04,
                0.045, 0.05, 0.06, 0.07, 0.08, 0.085
            ]
            # eps_arr = [0.05, 0.06, 0.07, 0.08]
        elif eta == 0.1:
            eps_arr = [
                0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06,
                0.07
            ]
            # eps_arr = [0.05, 0.06, 0.07, 0.075]
        elif eta == 0.05:
            eps_arr = [
                0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.04, 0.045, 0.05,
                0.055
            ]
            # eps_arr = [0.04, 0.045, 0.05, 0.055]
    elif disorder_t == "RF":
        if eta == 0.18:
            eps_arr = [
                0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095, 0.1, 0.11, 0.12,
                0.13, 0.14, 0.15, 0.16
            ]
            # eps_arr = [0.13, 0.14, 0.15, 0.16]

    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        if rescaled:
            chi_dis /= (L**2 * phi**2)
            chi_con /= (L**2 * phi**2)
        else:
            chi_dis /= (L**2)
            chi_con /= (L**2)
        chi_tot = chi_dis + chi_con
        fillstyle = "none"
        line, = axes[0].plot(
            L, chi_dis, "-o", fillstyle=fillstyle, lw=1, label="%g" % eps)
        c = line.get_c()
        axes[1].plot(
            L, chi_con, "-o", lw=1, label="%g" % eps, c=c, fillstyle=fillstyle)
        axes[2].plot(
            L, chi_tot, "-o", lw=1, label="%g" % eps, c=c, fillstyle=fillstyle)
        if rescaled:
            L_mid, slope = get_slope(L, phi)
            axes[3].plot(
                L_mid, slope, "s", fillstyle="none", c=c, label="%g" % eps)
        else:
            axes[3].plot(
                L,
                chi_con / chi_dis,
                "-o",
                lw=1,
                label="%g" % eps,
                fillstyle=fillstyle,
                c=c)

    if disorder_t == "RT":
        axes[2].legend(frameon=False, labelspacing=0.05, fontsize="small")
    elif disorder_t == "RF":
        axes[3].legend(frameon=False, labelspacing=0.05, fontsize="small")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    # axes[0][0].set_xlabel(r"$L$", fontsize="x-large")
    # axes[0][1].set_xlabel(r"$L$", fontsize="x-large")
    axes[0].set_xlabel(r"$L$", fontsize="x-large")
    axes[1].set_xlabel(r"$L$", fontsize="x-large")
    axes[2].set_xlabel(r"$L$", fontsize="x-large")
    axes[3].set_xlabel(r"$L$", fontsize="x-large")
    if rescaled:
        axes[0].set_ylabel(r"$R$", fontsize="x-large")
        axes[1].set_ylabel(r"$\chi_{\rm con}/L^2M^2$", fontsize="x-large")
        axes[2].set_ylabel(r"$\chi_{\rm tot}/L^2M^2$", fontsize="x-large")
        axes[3].set_ylabel(r"$\sigma$", fontsize="x-large")
    else:
        axes[0].set_ylabel(r"$\chi_{\rm dis}/L^2$", fontsize="x-large")
        axes[1].set_ylabel(r"$\chi_{\rm con}/L^2$", fontsize="x-large")
        axes[2].set_ylabel(r"$\chi_{\rm tot}/L^2$", fontsize="x-large")
        axes[3].set_ylabel(
            r"$\chi_{\rm con}/\chi_{\rm dis}$", fontsize="x-large")
    plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.98])

    if rescaled is False:
        if disorder_t == "RT" and eta == 0.18:
            x = [180, 256, 362, 512]
            y1 = np.array([4.1942e-05, 4.62696E-5, 5.461872e-05, 6.5631e-05])
            y2 = np.array([6.6418e-05, 6.97583E-5, 8.489967e-05, 0.00012228])
            axes[0].plot(x, y1, ">", fillstyle="none", c="b")
            # axes[0].plot(x, y2, "s", fillstyle="none", c="tab:pink")
            y3 = np.array(
                [0.000133599, 0.00010373, 8.290644e-05, 6.9771884e-05])
            y4 = np.array(
                [0.000155696, 0.000121197, 0.000100319, 9.381793e-05])
            axes[1].plot(x, y3, ">", fillstyle="none", c="b")
            # axes[1].plot(x, y4, "s", fillstyle="none", c="b")
            axes[2].plot(x, y1 + y3, ">", fillstyle="none", c="b")
            # axes[2].plot(x, y2 + y4, "s", fillstyle="none", c="tab:red")

            chi_replica = np.array([1.718E-5, 1.8275E-5, 2.2207E-5, 3.2122E-5])
            axes[0].plot(x, chi_replica, "ks", fillstyle="full")
        if disorder_t == "RT":
            if eta == 0.18:
                add_line(
                    axes[0], 0, 0.1, 1, -1.2, scale="log", c="tab:green", lw=2)
                # add_line(
                #     axes[0], 0.3, 1, 1, -3.5, scale="log", c="tab:blue", lw=2)
                add_line(
                    axes[1], 0.28, 1, 1, -2, scale="log", c="tab:orange", lw=2)
                add_line(
                    axes[1],
                    0,
                    0.55,
                    1,
                    -1.2,
                    scale="log",
                    c="tab:green",
                    lw=2)
                add_line(axes[0], 0, 0.5, 1, 2 / 3, scale="log", lw=2)
                add_line(axes[0], 0, 0.25, 1, 2 / 3, scale="log", lw=2)
                add_line(axes[0], 0, 0.05, 1, 2 / 3, scale="log", lw=2)
                add_line(axes[1], 0, 0.45, 1, -0.22, scale="log", c="tab:pink")

            elif eta == 0.10:
                add_line(axes[0], 0, 0.05, 1, 2 / 3, scale="log", lw=2)
                add_line(axes[0], 0, 0.25, 1, 2 / 3, scale="log", lw=2)
                add_line(axes[0], 0, 0.4, 1, 0.32, scale="log", c="k", lw=2)
                add_line(axes[0], 0, 0.5, 1, 0.32, scale="log", c="k", lw=2)

        elif disorder_t == "RF":
            add_line(axes[0], 0, 0.08, 1, 2 / 3, scale="log", lw=2)
            add_line(axes[0], 0, 0.14, 1, 2 / 3, scale="log", lw=2)
            add_line(axes[0], 0, 0.2, 1, 2 / 3, scale="log", lw=2)
            add_line(axes[0], 0, 0.285, 1, 2 / 3, scale="log", lw=2)
            add_line(axes[0], 0, 0.34, 1, 2 / 3, scale="log", lw=2)
            add_line(axes[0], 0, 0.38, 1, 2 / 3, scale="log", lw=2)
            # add_line(axes[0], 0, 0.4, 1, 0.44, scale="log", lw=2, c="k")
            add_line(axes[1], 0.5, 0.31, 1, -0.22, scale="log", c="tab:brown")
            add_line(axes[1], 0.5, 0.25, 1, -0.31, scale="log", c="tab:purple")

        # axes[1].axhline(1e-4, c="k", linestyle="dotted")
    else:
        axes[3].axhline(0.04, c="k", linestyle="dotted")
        # add_line(axes[0], 0.2, 1, 1, -1.5, scale="log", c="b")
        pass

    if disorder_t == "RF":
        eps_arr = [0.09, 0.08]
        c_dict = {0.09: "tab:brown", 0.08: "tab:purple"}
    else:
        eps_arr = [0.035]
        c_dict = {0.035: "tab:pink"}
    for eps in eps_arr:
        L, chi_dis, chi_con, M = get_chi_M(eps, eta, disorder_t)
        c = c_dict[eps]
        if rescaled:
            axes[0].plot(L, chi_dis / (L * M)**2, "^", c=c, fillstyle="none")
            axes[1].plot(L, chi_con / (L * M)**2, "^", c=c, fillstyle="none")
            axes[2].plot(
                L, (chi_con + chi_dis) / (L * M)**2,
                "^",
                c=c,
                fillstyle="none")
            x, y = get_slope(L, M)
            axes[3].plot(x, y, "^", fillstyle="none", c=c)
        else:
            axes[0].plot(L, chi_dis / L**2, "^", c=c, fillstyle="none")
            axes[1].plot(L, chi_con / L**2, "^", c=c, fillstyle="none")
            axes[2].plot(
                L, (chi_con + chi_dis) / L**2, "^", c=c, fillstyle="none")

    if disorder_t == "RT":
        title = r"RS: $\eta=%g$" % eta
    else:
        title = r"%s: $\eta=%g$" % (disorder_t, eta)
    plt.suptitle(title, y=0.998, fontsize="x-large")
    plt.show()
    plt.close()


def plot_5(eta, rescaled=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))

    eps_arr = [0.01, 0.02, 0.03, 0.035, 0.04, 0.045]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, "RT")
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        if rescaled:
            chi_dis /= (L**2 * phi**2)
        else:
            chi_dis /= (L**2)
        mask = L < 1448
        ax.plot(
            L[mask], chi_dis[mask], ":o", label="%g" % eps, fillstyle="none")

    eps_arr = [0.04, 0.06, 0.07, 0.08, 0.1]
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, "RF")
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        if rescaled:
            chi_dis /= (L**2 * phi**2)
        else:
            chi_dis /= (L**2)
        mask = L < 1448
        ax.plot(
            L[mask], chi_dis[mask], "--s", label="%g" % eps, fillstyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if rescaled:
        ax.set_ylabel(r"$R$", fontsize="x-large")
    else:
        ax.set_ylabel(r"$\chi_{\rm dis}/L^2$", fontsize="x-large")
    ax.set_xlabel(r"$L$", fontsize="x-large")
    slope = 0.65
    add_line(
        ax, 0, 0.2, 1, slope, r"$L^{%g}$" % slope, xl=0.5, yl=0.3, scale="log")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_chi_peak(eta, disorder_t="RT"):
    from fit import find_peak_polyfit
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.tick_params(direction='in', which="both")
    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            eps_arr = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085]
        elif eta == 0.1:
            eps_arr = [0.04, 0.045, 0.05, 0.06, 0.07]
        elif eta == 0.05:
            eps_arr = [0.04, 0.045, 0.05, 0.055]
    elif disorder_t == "RF":
        if eta == 0.18:
            eps_arr = [0.13, 0.14, 0.15, 0.16]

    lines = []
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        chi_dis /= L**2.2
        # chi_dis_over_L2 = chi_dis / L**2
        # L, chi_con = chi_con_dict[eps]
        # chi_con_over_L2 = chi_con / L**2
        # chi_tot = chi_dis + chi_con

        fillstyle = "none"

        line, = ax.plot(L, chi_dis, "o", fillstyle=fillstyle, label="%g" % eps)
        c = line.get_c()
        L_peak, chi_peak, L_err, chi_err, coeff = find_peak_polyfit(
            L.copy(), chi_dis.copy(), "log", "log")
        if eps != 0.05:
            ax.plot(L_peak, chi_peak, "s", c=c, ms=8)
        x = np.linspace(L.min(), L.max(), 1000)
        y = np.exp(np.polyval(coeff, np.log(x)))
        ax.plot(x, y, "--", c=c)
        print("%g\t%f\t%f" % (eps, L_peak, chi_peak))
        # L_peak, chi_peak, L_err, chi_err, c = find_peak_polyfit(
        #     L, chi_dis_over_L2, "log", "log")
        # ax.plot(L_peak, chi_peak, "p", c=c)
        lines.append(line)
    ax.legend(title=r"$\epsilon=$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$", fontsize="x-large")
    ax.set_ylabel(r"$\chi_{\rm dis}$", fontsize="x-large")
    plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.98])

    if disorder_t == "RT":
        title = r"RS: $\eta=%g$" % eta
    else:
        title = r"%s: $\eta=%g$" % (disorder_t, eta)
    plt.suptitle(title, y=0.998, fontsize="x-large")
    plt.show()
    plt.close()


def plot_M(eta, disorder_t="RT"):
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)
    ax1.tick_params(direction='in', which="both")
    ax2.tick_params(direction='in', which="both")

    phi_dict, chi_con_dict, chi_dis_dict = get_data(eta, disorder_t)
    if disorder_t == "RT":
        if eta == 0.18:
            eps_arr = [
                0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.035, 0.037, 0.04,
                0.045, 0.05
            ]
        elif eta == 0.1:
            eps_arr = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
        elif eta == 0.05:
            eps_arr = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.04]
    elif disorder_t == "RF":
        if eta == 0.18:
            eps_arr = [
                0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095, 0.1, 0.11, 0.12
            ]

    for eps in eps_arr:
        L, phi = phi_dict[eps]
        mask = L <= 4096
        L = L[mask]
        phi = phi[mask]
        line, = ax1.plot(L, phi, "-o", label="$%g$" % eps, fillstyle="none")
        ax2.plot(
            L, phi, "-o", c=line.get_c(), label="$%g$" % eps, fillstyle="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel(r"$L$", fontsize="large")
    ax1.set_ylabel(r"$M$", fontsize="large")
    ax2.set_xlabel(r"$L$", fontsize="large")
    ax2.set_ylabel(r"$M$", fontsize="large")
    ax1.legend()
    if disorder_t == "RT":
        title = "RS: " + r"$\eta=%g$" % eta
        if eta == 0.18:
            ax2.set_xlim(30)
            ax2.set_ylim(0.68, 0.82)
            x = [180, 256, 362, 512]
            y1 = [0.797964, 0.79008, 0.7823912, 0.773611]
            # y2 = [0.795017, 0.78680, 0.7789784, 0.768422]
            ax2.plot(x, y1, ">", fillstyle="none", c="r")
            # ax2.plot(x, y2, "s", fillstyle="none", c="tab:pink")
            add_line(ax2, 0.6, 0.68, 0.95, -0.04, scale="log", c="tab:pink")
            L, chi_dis, chi_con, M = get_chi_M(0.035, eta, disorder_t)
            ax2.plot(L, M, "^", fillstyle="none", c="tab:pink")

    elif disorder_t == "RF":
        title = "RF: " + r"$\eta=%g$" % eta
        if eta == 0.18:
            ax2.set_ylim(0.71, 0.84)
            add_line(ax2, 0.6, 0.55, 0.99, -0.04, scale="log", c="tab:brown")
            L, chi_dis, chi_con, M = get_chi_M(0.09, eta, disorder_t)
            ax1.plot(L, M, "^", fillstyle="none", c="tab:brown")
            ax2.plot(L, M, "^", fillstyle="none", c="tab:brown")
            L, chi_dis, chi_con, M = get_chi_M(0.08, eta, disorder_t)
            ax1.plot(L, M, "^", fillstyle="none", c="tab:purple")
            ax2.plot(L, M, "^", fillstyle="none", c="tab:purple")

    plt.suptitle(title, fontsize="x-large")
    plt.show()
    plt.close()


def plot_chi_dis_over_chi_con_square():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True)
    axes[0].tick_params(direction='in', which="both")
    axes[1].tick_params(direction='in', which="both")

    phi_dict, chi_con_dict, chi_dis_dict = get_data(0.18, "RT")
    eps_arr = [
        0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06,
        0.07, 0.08, 0.085
    ]

    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        y = chi_dis / chi_con**2
        if eps < 0.045:
            axes[0].plot(L, y, "-o", label="%g" % eps)
        else:
            axes[0].plot(L, y, "-s", label="%g" % eps, fillstyle="none")

    phi_dict, chi_con_dict, chi_dis_dict = get_data(0.18, "RF")
    eps_arr = [
        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
        0.16
    ]
    for eps in eps_arr:
        L, phi = phi_dict[eps]
        L, chi_dis = chi_dis_dict[eps]
        L, chi_con = chi_con_dict[eps]
        y = chi_dis / chi_con**2
        if eps < 0.1:
            axes[1].plot(L, y, "-o", label="%g" % eps)
        else:
            axes[1].plot(L, y, "-s", label="%g" % eps, fillstyle="none")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")

    axes[0].set_xlabel(r"$L$", fontsize="x-large")
    axes[1].set_xlabel(r"$L$", fontsize="x-large")
    axes[0].set_ylabel(
        r"$\chi_{\rm dis}/\chi_{\rm con}^2$", fontsize="x-large")
    axes[1].set_ylabel(
        r"$\chi_{\rm dis}/\chi_{\rm con}^2$", fontsize="x-large")

    # add_line(axes[0], 0, 0.5, 1, -3, scale="log", lw=2)
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title(r"RS: $\eta=0.18$", fontsize="x-large")
    axes[1].set_title(r"RF: $\eta=0.18$", fontsize="x-large")
    plt.tight_layout(rect=[-0.01, -0.025, 1.01, 0.98])
    plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_1()
    plot_4(0.18, "RT", rescaled=False)
    # plot_5(0.18, False)
    # plot_chi_peak(0.18, "RT")
    # plot_M(0.18, "RT")
    # plot_chi_dis_over_chi_con_square()
