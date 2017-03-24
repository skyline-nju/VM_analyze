""" Plot rho_gas vs. phi. """

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from read_npz import read_matched_file, eq_Lx_and_nb, time_average


def get_data(nb, Lx=None):
    if Lx is None:
        os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps0")
        if nb == 2:
            Lxs = range(280, 460, 20)
        elif nb == 3:
            Lxs = range(500, 700, 20)
        elif nb == 4:
            Lxs = range(680, 940, 20)
        elif nb == 5:
            Lxs = range(820, 1000, 20)
        else:
            print("nb is too large")
            return
        phi = []
        peak = []
        Lx = []
        v = []

        for i, Lx0 in enumerate(Lxs):
            dictLSN = read_matched_file({"Lx": Lx0})
            seed = sorted(dictLSN[Lx0].keys())[0]
            if nb in dictLSN[Lx0][seed]:
                phi.append(dictLSN[Lx0][seed][nb]["mean_phi"])
                peak.append(dictLSN[Lx0][seed][nb]["ave_peak"])
                v.append(dictLSN[Lx0][seed][nb]["mean_v"])
                Lx.append(Lx0)
        phi = np.array(phi)
        peak = np.array(peak)
        v = np.array(v)
        Lx = np.array(Lx)
        return phi, peak, v, Lx
    else:
        if 400 <= Lx <= 480:
            os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\uniband")
        else:
            os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps20")
        dict0 = read_matched_file({"Lx": Lx})
        phi = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "mean_phi", dictLSN=dict0)])
        peak = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "ave_peak", dictLSN=dict0)])
        v = np.array(
            [i for i in eq_Lx_and_nb(Lx, nb, "mean_v", dictLSN=dict0)])
        return phi, peak, v


def get_data_varied_eta(Lx, nb=2, Ly=200, eps=0, rho0=1):
    os.chdir(r"E:\data\random_torque\bands\Lx\snapshot\varied_eta")
    files = glob.glob("mb_*.%d.%d.%d.*.npz" % (eps, Lx, Ly))
    phi = defaultdict(float)
    c = defaultdict(float)
    rho_exc = defaultdict(float)
    weight = defaultdict(float)
    for file in files:
        eta = int(file.replace("mb_", "").split(".")[0]) / 1000
        data0 = time_average(file)
        if nb in data0:
            rate = data0[nb]["rate"]
            phi[eta] += data0[nb]["mean_phi"] * rate
            c[eta] += data0[nb]["mean_v"] * rate
            peak = data0[nb]["ave_peak"]
            rho_exc[eta] += (rho0 - np.mean(peak[190:200])) * rate
            weight[eta] += rate
    eta = np.array(sorted(phi.keys()))
    phi = np.array([phi[key] / weight[key] for key in eta])
    rho_exc = np.array([rho_exc[key] / weight[key] for key in eta])
    c = np.array([c[key] / weight[key] for key in eta])
    return eta, phi, rho_exc, c


def plot_eq_nb_and_Lx(nb, Lx):
    """ Plot rho_gas vs phi with fixed nb and Lx. """

    phi, peak, v = get_data(nb, Lx)
    cList = plt.cm.jet(
        [(i - phi.min()) / (phi.max() - phi.min()) for i in phi])
    plt.subplot(121)
    x = np.arange(Lx) + 0.5
    for i, pk in enumerate(peak):
        plt.plot(x, pk, c=cList[i])

    plt.subplot(122)
    for xmax in range(200, 250, 10):
        rho_gas = np.mean(peak[:, 190:xmax], axis=1)
        rho_exc = 1 - rho_gas
        z = np.polyfit(rho_exc, phi, 1)
        plt.plot(
            rho_exc,
            phi,
            "o",
            label=r"$x_{max}=%d, \rm{slope}=%g$" % (xmax, z[0]))

    plt.legend(loc="best")
    plt.show()
    plt.close()
    print("velocity: mean=%g, std=%g" % (v.mean(), v.std()))


def case_eps0_eq_nb(nb):
    """ Plot rho_gas vs phi with fixed nb for disorder free case."""

    def get_rho_gas(peak, min=190, max=210):
        rho_gas = np.zeros(peak.size)
        for i, pk in enumerate(peak):
            rho_gas[i] = np.mean(pk[min:max])
        return rho_gas

    phi, peak, v, Lx = get_data(nb)
    plt.subplot(121)
    for i, pk in enumerate(peak):
        x = np.arange(Lx[i]) + 0.5
        plt.plot(x, pk)

    plt.subplot(122)
    xmin = 190
    for xmax in range(200, 240, 5):
        rho_gas = get_rho_gas(peak, xmin, xmax)
        rho_exc = 1 - rho_gas
        z = np.polyfit(rho_exc, phi, 1)
        print(z[0] * 0.5, np.mean(v))
        plt.plot(rho_exc, phi, "-o", label=r"$x\in[%d: %d]$" % (xmin, xmax))
    plt.legend(loc="best")
    plt.show()
    plt.close()

    plt.plot(Lx, v, "-s")
    plt.show()


def plot_varied_eta(v0=0.5):
    def panel4(Lx, marker):
        print("L = %d" % Lx)
        eta, phi, rho_exc, c = get_data_varied_eta(Lx)
        v = phi * v0
        u = rho_exc * c

        # subpolot(221)
        z = np.polyfit(eta, rho_exc, 1)
        print(z)
        ax1.plot(
            eta,
            rho_exc,
            marker,
            label=r"$L_x=%d,{\rm slope}=%.4f$" % (Lx, z[0]))

        # subplot(222)
        z = np.polyfit(eta, c, 1)
        print(z)
        ax2.plot(
            eta, c, marker, label=r"$L_x=%d,{\rm slope}=%.4f$" % (Lx, z[0]))

        # subplot(223)
        z = np.polyfit(eta, v, 1)
        print(z)
        ax3.plot(
            eta, v, marker, label=r"$L_x=%d,{\rm slope}=%.4f$" % (Lx, z[0]))

        # subplot(224)
        z = np.polyfit(u, v, 1)
        print(z)
        ax4.plot(u, v, marker, label=r"$L_x=%d,{\rm slope}=%.4f$" % (Lx, z[0]))

    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(15, 4))
    ax1, ax2, ax3, ax4 = axes.flat

    panel4(360, "-o")
    panel4(440, "-s")

    # set legend
    ax1.legend(loc=(0.01, 0.01), fontsize=11.5, labelspacing=0, borderpad=0.1)
    ax2.legend(loc=(0.01, 0.01), fontsize=11.5, labelspacing=0, borderpad=0.1)
    ax3.legend(loc=(0.01, 0.01), fontsize=11.5, labelspacing=0, borderpad=0.1)
    ax4.legend(loc=(0.01, 0.77), fontsize=11.5, labelspacing=0, borderpad=0.1)

    # set xlabel, ylabel
    ax1.set_ylim(0.5825, 0.6075)
    xlabel = r"$\eta$"
    ylabel = r"$\rho_0-\rho_{\rm gas}$"
    ax1.text(0.93, 0.02, xlabel, transform=ax1.transAxes, fontsize="xx-large")
    ax1.text(0.01, 0.94, ylabel, transform=ax1.transAxes, fontsize="xx-large")

    ax2.set_ylim(0.4088, 0.412)
    ylabel = r"$c$"
    ax2.text(0.93, 0.02, xlabel, transform=ax2.transAxes, fontsize="xx-large")
    ax2.text(0.01, 0.94, ylabel, transform=ax2.transAxes, fontsize="xx-large")

    ax3.set_ylim(0.239, 0.251)
    ylabel = r"$|{\bf v}|$"
    ax3.text(0.93, 0.02, xlabel, transform=ax3.transAxes, fontsize="xx-large")
    ax3.text(0.01, 0.94, ylabel, transform=ax3.transAxes, fontsize="xx-large")

    xlabel = r"$c(\rho_0-\rho_{\rm gas})$"
    ylabel = r"$|{\bf v}|$"
    ax4.text(0.55, 0.02, xlabel, transform=ax4.transAxes, fontsize="xx-large")
    ax4.text(0.01, 0.94, ylabel, transform=ax4.transAxes, fontsize="xx-large")

    # set axis number
    ax4.set_ylim(ymax=0.251)
    set_subplot_number(axes, 0.89, 0.94, fontsize="x-large")

    add_slope_line(ax1, -3, 0.5, 0.3, 0.2, 0.4)
    add_slope_line(ax2, -0.4, 0.5, 0.3, 0.2, 0.4)
    add_slope_line(ax3, -1.5, 0.5, 0.3, 0.2, 0.4)
    add_slope_line(ax4, 1, 0.6, 0.4)

    title = r"$\epsilon=0, \rho_0=1, L_y=200, n_b=2$"
    plt.suptitle(title, y=0.99, fontsize="xx-large", color="b")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def set_subplot_number(axes, x, y, number=None, fontsize="x-larrge"):
    """ set the number of subplots. """
    order = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    bbox = dict(edgecolor="k", fill=False)
    if number is None or isinstance(number, list):
        for i, ax in enumerate(axes.flat):
            ax.text(
                x,
                y,
                order[i],
                transform=ax.transAxes,
                bbox=bbox,
                fontsize=fontsize)
    else:
        ax = axes
        ax.text(
            x, y, number, transform=ax.transAxes, bbox=bbox, fontsize=fontsize)


def plot_varied_Lx(v0=0.5):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    ax1, ax2, ax3, ax4 = axes.flat
    phi, peak, c, Lx = get_data(2)
    rho_exc = get_rho_exc(peak)
    v = v0 * phi
    u = rho_exc * c

    ax1.plot(Lx, rho_exc, "-o")
    print(np.polyfit(Lx[4:], rho_exc[4:], 1))
    ax2.plot(Lx, c, "-o")
    print(np.polyfit(Lx, c, 1))
    ax3.plot(Lx, v, "-o")
    print(np.polyfit(Lx, v, 1))
    ax4.plot(u, v, "-o")

    # set xlabel, ylabel
    ax1.set_ylim(0.6, 0.611)
    xlabel = r"$L_x$"
    ylabel = r"$\rho_0-\rho_{\rm gas}$"
    ax1.text(0.85, 0.02, xlabel, transform=ax1.transAxes, fontsize="xx-large")
    ax1.text(0.01, 0.94, ylabel, transform=ax1.transAxes, fontsize="xx-large")

    ax2.set_ylim(ymax=0.41185)
    ylabel = r"$c$"
    ax2.text(0.91, 0.02, xlabel, transform=ax2.transAxes, fontsize="xx-large")
    ax2.text(0.01, 0.94, ylabel, transform=ax2.transAxes, fontsize="xx-large")

    ax3.set_ylim(0.2475, 0.2520)
    ylabel = r"$|{\bf v}|$"
    ax3.text(0.85, 0.02, xlabel, transform=ax3.transAxes, fontsize="xx-large")
    ax3.text(0.01, 0.94, ylabel, transform=ax3.transAxes, fontsize="xx-large")

    xlabel = r"$c(\rho_0-\rho_{\rm gas})$"
    ylabel = r"$|{\bf v}|$"
    ax4.text(0.01, 0.94, ylabel, transform=ax4.transAxes, fontsize="xx-large")
    ax4.text(0.55, 0.02, xlabel, transform=ax4.transAxes, fontsize="xx-large")

    # set number of subplots
    ax4.set_ylim(ymax=0.252)
    set_subplot_number(axes, 0.89, 0.94, fontsize="x-large")

    # add lines to show slope
    label = r"${\rm slope}=-5.9\times 10^{-5}$"
    add_slope_line(ax1, -5.9e-5, 0.5, 0.3, 0.2, 0.4, label=label)
    label = r"${\rm slope}=-2.4\times 10^{-5}$"
    add_slope_line(ax3, -2.4e-5, 0.5, 0.3, 0.2, 0.4, label=label)
    add_slope_line(ax4, 1, 0.6, 0.4)

    title = r"$\eta=0.35, \epsilon=0, \rho_0=1, L_y=200, n_b=2$"
    plt.suptitle(title, y=0.99, fontsize="xx-large", color="b")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()
    pass


def get_rho_exc(peak, xmin=190, xmax=200, rho0=1):
    rho_gas = np.array([np.mean(pk[xmin:xmax]) for pk in peak])
    rho_exc = rho0 - rho_gas
    return rho_exc


def slope_line(ax, slope, x0, y0, x1, label=None, xl=None, yl=None, deg=45):
    c = "#7f7f7f"
    x = np.linspace(x0, x1, 100)
    y = y0 + slope * (x - x0)
    ax.plot(x, y, "-.", lw=3, c=c)
    if label is not None:
        ax.text(xl, yl, label, transform=ax.transAxes, rotation=deg, color=c)


def add_slope_line(ax,
                   slope,
                   xc,
                   yc,
                   xl=None,
                   yl=None,
                   label=None,
                   fontsize="large"):
    c = "#7f7f7f"
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    slope_new = slope * (xmax - xmin) / (ymax - ymin)
    x = np.linspace(0, 1, 100)
    y = slope_new * (x - xc) + yc
    ax.plot(x, y, "-.", transform=ax.transAxes, c=c)
    deg = np.arctan(slope_new) * 180 / np.pi
    if xl is None:
        xl = xc
    if yl is None:
        yl = yc
    if label is None:
        label = r"${\rm slope}=%g$" % slope
    ax.text(
        xl,
        yl,
        label,
        transform=ax.transAxes,
        rotation=deg,
        color=c,
        fontsize=fontsize)


def plot_eps20(v0=0.5):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

    # subplot(121)
    Lxs = np.array([400, 420, 440, 460, 480])
    # Lxs = np.array([480])
    u_m = np.zeros(Lxs.size)
    v_m = np.zeros(Lxs.size)
    slope_m = np.zeros(Lxs.size)
    for i, Lx in enumerate(Lxs):
        phi, peak, c = get_data(2, Lx)
        rho_exc = get_rho_exc(peak, xmin=190, xmax=200)
        u = rho_exc * c
        v = phi * v0
        ax1.plot(u, v, "o", ms=3, label=r"$L_x=%d$" % Lx)
        slope_m[i] = np.polyfit(u, v, 1)[0]
        u_m[i] = u.mean()
        v_m[i] = np.mean(v)

    # subplot(122)
    x = np.linspace(0.216, 0.222, 100)
    ax2.set_xlim(0.217, 0.221)
    ax2.set_ylim(0.2185, 0.2225)
    for i, slope in enumerate(slope_m):
        y = slope * (x - u_m[i]) + v_m[i]
        line, = ax2.plot(x, y)
        ax2.plot(
            u_m[i],
            v_m[i],
            "s",
            color=line.get_c(),
            label=r"$L_x=%d$" % Lxs[i])
        print("Lx=%d, slope=%g" % (Lxs[i], slope))
    z = np.polyfit(u_m, v_m, 1)
    y = z[0] * x + z[1]
    ax2.plot(x, y, "k--")
    print("mean slope = %g" % z[0])

    add_slope_line(ax1, 1, 0.6, 0.4, 0.55, 0.45)
    add_slope_line(ax2, 1, 0.65, 0.35, 0.6, 0.4)
    add_slope_line(ax2, 1.16, 0.4, 0.6, 0.3, 0.8)

    # set axis label
    xlabel = r"$|{\bf v}|$"
    ylabel = r"$c\left (\rho_0 -\rho_{\rm gas}\right )$"
    ax1.text(0.01, 0.94, xlabel, transform=ax1.transAxes, fontsize="large")
    ax2.text(0.01, 0.94, xlabel, transform=ax2.transAxes, fontsize="large")
    ax1.text(0.68, 0.02, ylabel, transform=ax1.transAxes, fontsize="large")
    ax2.text(0.68, 0.02, ylabel, transform=ax2.transAxes, fontsize="large")

    # set legend
    ax1.legend(loc=(0.01, 0.55), fontsize="small")
    ax2.legend(loc=(0.01, 0.55), fontsize="small")

    ax1.set_title(r"(a)$\epsilon=0.02, n_b=2$")
    ax2.set_title(r"(b)$\epsilon=0.02, n_b=2$, sample average")

    plt.tight_layout()
    plt.show()
    # plt.savefig(r"E:\report\quenched_disorder\report\fig\rho_exc_20.pdf")
    plt.close()


if __name__ == "__main__":
    # plot_varied_eta()
    plot_varied_Lx()
