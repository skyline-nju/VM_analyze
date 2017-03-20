""" Plot rho_gas vs. phi. """

import os
import numpy as np
import matplotlib.pyplot as plt
from read_npz import read_matched_file, eq_Lx_and_nb


def get_data(nb, Lx=None):
    if Lx is None:
        os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\eps0")
        if nb == 2:
            Lxs = range(320, 480, 20)
        elif nb == 3:
            Lxs = range(480, 740, 20)
        elif nb == 4:
            Lxs = range(680, 940, 20)
        elif nb == 5:
            Lxs = range(800, 1020, 20)
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


def four_panels():
    def get_rho_exc(peak, xmin=190, xmax=210, rho0=1):
        rho_gas = np.array([np.mean(pk[xmin:xmax]) for pk in peak])
        rho_exc = rho0 - rho_gas
        return rho_exc

    def slope_line(ax, slope, x0, y0, x1, label=None, xl=None, yl=None,
                   deg=45):
        c = "#7f7f7f"
        x = np.linspace(x0, x1, 100)
        y = y0 + slope * (x - x0)
        ax.plot(x, y, "-.", lw=3, c=c)
        if label is not None:
            ax.text(
                xl, yl, label, transform=ax.transAxes, rotation=deg, color=c)

    v0 = 0.5
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes.flat

    # subplot(221)
    Lxs = np.array([400, 420, 440, 460, 480])
    slope = np.zeros(Lxs.size)
    u_m = np.zeros(Lxs.size)
    rho_exc_m = np.zeros(Lxs.size)
    v_m = np.zeros(Lxs.size)
    for i, Lx in enumerate(Lxs):
        phi, peak, v = get_data(2, Lx)
        rho_exc = get_rho_exc(peak, xmin=190, xmax=200)
        u = phi * v0
        z = np.polyfit(rho_exc, u, 1)
        slope[i] = z[0]
        u_m[i] = np.mean(u)
        v_m[i] = np.mean(v)
        rho_exc_m[i] = np.mean(rho_exc)
        ax1.plot(rho_exc, u, "o", label=r"$L_x=%d$" % (Lx))
        print("Lx=%d, v=%f" % (Lx, v.mean()))

    llabel = r"$|{\bf v}| \propto c \left (\rho_0-\rho_{\rm gas}\right )$"
    slope_line(ax1, np.mean(v_m), 0.5225, 0.2135, 0.545, llabel, 0.5, 0.45, 50)

    ax1.set_title(r"(a)$\epsilon=0.02, n_b=2$")
    ax1.legend(loc=(0.01, 0.5))

    # subplot(222)
    xmin, xmax = 0.529, 0.538
    x = np.linspace(xmin, xmax, 50)
    c_list = []
    for i in range(slope.size):
        y = u_m[i] + slope[i] * (x - rho_exc_m[i])
        line, = ax2.plot(x, y)
        c_list.append(line.get_c())
    z = np.polyfit(rho_exc_m, u_m, 1)
    ym = z[0] * x + z[1]
    ax2.plot(x, ym, "k:", lw=3, label="linear fit")
    for i in range(slope.size):
        ax2.plot(
            rho_exc_m[i], u_m[i], "s", c=c_list[i], label=r"$L_x=%d$" % Lxs[i])

    slope_line(ax2, np.mean(v_m), 0.531, ym.min(), 0.538, llabel, 0.5, 0.4, 47)
    ax2.axis("tight")
    ax2.set_title(r"(b)$\epsilon=0.02, n_b=2$, sample average")
    ax2.legend(loc=(0.01, 0.48))

    # subplot(223)
    Lxs = [440, 660, 880]
    for i, Lx in enumerate(Lxs):
        nb = Lx // 220
        phi, peak, v = get_data(nb, Lx)
        rho_exc = get_rho_exc(peak, xmin=190, xmax=200)
        ax3.plot(
            rho_exc, phi * v0, "o", label=r"$L_x=%d, n_b=%d$" % (Lx, nb))
    ax3.set_xlim(0.526)
    ax3.set_ylim(0.2175)
    slope_line(ax3, np.mean(v_m), 0.53, 0.218, 0.541, llabel, 0.5, 0.45, 51)
    ax3.legend(loc=(0.01, 0.62))
    ax3.set_title(r"(c)$\epsilon=0.02, L_r/n_b\lambda=0$")

    # subplot(224)
    phi, peak, v, Lx = get_data(2)
    rho_exc = get_rho_exc(peak, xmax=200)
    u = phi * v0
    z = np.polyfit(rho_exc, u, 1)
    x = np.linspace(rho_exc.min() - 0.0005, rho_exc.max() + 0.0005, 50)
    y = x * z[0] + z[1]
    ax4.plot(x, y, "k:", label="liner fit", lw=2)
    for i in range(u.size):
        ax4.plot(rho_exc[i], u[i], "o", label=r"$L_x=%d$" % (Lx[i]))

    slope_line(ax4, np.mean(v), 0.6015, y.min(), 0.608, llabel, 0.5, 0.45, 48)
    ax4.set_title(r"(d)$\epsilon=0, n_b=2$")
    ax4.legend(loc=(0.01, 0.4), labelspacing=0.2)

    # set labels of x, y axis
    xlabel = r"$\rho_0-\rho_{\rm gas}$"
    ylabel = r"$|{\bf v}|$"
    for ax in axes.flat:
        ax.text(0.70, 0.03, xlabel, fontsize=14, transform=ax.transAxes)
        ax.text(0.01, 0.94, ylabel, fontsize=14, transform=ax.transAxes)
    plt.tight_layout()

    # plt.savefig(r"E:\report\quenched_disorder\report\fig\rho_exc.eps")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # case_eps0_eq_nb(2)
    # plot_eq_nb_and_Lx(2, 440)
    four_panels()
