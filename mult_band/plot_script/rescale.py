# plot phi against Lr/nb*lambda

import os
import numpy as np
import matplotlib.pyplot as plt
from read_npz import sample_average
from matplotlib.markers import MarkerStyle


def read(**kwargs):
    Lx = []
    phi = []
    rate = []
    nb = []
    if "path" in kwargs:
        os.chdir(kwargs["path"])
        eps = kwargs["eps"]
        for Lx0 in kwargs["Lx"]:
            try:
                phi_dict, rate_dict = sample_average(Lx0, eps, rate_min=0.3)
                for nb0 in phi_dict:
                    phi.append(phi_dict[nb0])
                    Lx.append(Lx0)
                    rate.append(rate_dict[nb0])
                    nb.append(nb0)
            except RuntimeError:
                print("Error at Lx=%d" % Lx0)

    else:
        file = kwargs["file"]
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                s = line.replace("\n", "").split("\t")
                if len(s) == 4:
                    Lx.append(int(s[0]))
                    nb.append(int(s[1]))
                    phi.append(float(s[2]))
                    rate.append(float(s[3]))
                else:
                    Lx.append(int(s[0]))
                    nb.append(int(s[1]))
                    phi.append(float(s[3]))
                    rate.append(float(s[4]))

    Lx = np.array(Lx)
    phi = np.array(phi)
    rate = np.array(rate)
    nb = np.array(nb)
    return Lx, phi, rate, nb


def rescale(ax, Lx, phi, rate, nb, lamb, Ly, dLx, eta, eps, rho0=1):
    Lr = Lx - nb * lamb
    l_rescaled = Lr / (nb * lamb)
    circle, = ax.plot(l_rescaled, phi, "o")

    z = np.polyfit(l_rescaled, phi, 1)
    x = np.linspace(l_rescaled.min(), l_rescaled.max(), 100)
    y = x * z[0] + z[1]
    line, = ax.plot(x, y, "--")
    ax.text(
        0.05,
        0.05,
        r"$a=%f$" % z[0],
        transform=ax.transAxes,
        color=line.get_c(),
        fontsize="large")
    ax.text(
        0.05,
        0.15,
        r"$\Phi_c=%f$" % z[1],
        transform=ax.transAxes,
        color=line.get_c(),
        fontsize="large")
    ax.text(
        0.98,
        0.82,
        r"$\eta=%g, \epsilon=%g, \rho_0=%g$" % (eta, eps, rho0),
        horizontalalignment="right",
        fontsize="large",
        color=circle.get_c(),
        transform=ax.transAxes)
    ax.text(
        0.98,
        0.72,
        r"$L_y=%d, \Delta L_x=%d$" % (Ly, dLx),
        fontsize="large",
        horizontalalignment="right",
        color=circle.get_c(),
        transform=ax.transAxes)
    ax.text(
        0.98,
        0.62,
        r"$\lambda=%d$" % (lamb),
        fontsize="large",
        horizontalalignment="right",
        color=circle.get_c(),
        transform=ax.transAxes)
    ax.set_ylabel(r"$\Phi$", fontsize="x-large")
    ax.set_xlabel(r"$L_r/n_b\lambda$", fontsize="x-large")


def rescale_4_panel():
    file1 = drive + r"/data/random_torque/bands/Lx/350_0.dat"
    path2 = drive + r"/data/random_torque/bands/Lx/snapshot\eps20"
    file3 = drive + r"/data/random_torque/bands/Lx/old/400_0.dat"
    file4 = drive + r"/data/random_torque/bands/Lx/old/400_20.dat"

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 7))

    # subplot(221)
    Lx, phi, rate, nb = read(file=file1)
    rescale(axes[0][0], Lx, phi, rate, nb, 180, 200, 20, 0.35, 0)

    # subplot(222)
    Lx, phi, rate, nb = read(path=path2, Lx=range(300, 1020, 20), eps=20)
    rescale(axes[0][1], Lx, phi, rate, nb, 220, 200, 20, 0.35, 0.02)

    # subplot(223)
    Lx, phi, rate, nb = read(file=file3)
    rescale(axes[1][0], Lx, phi, rate, nb, 400, 100, 100, 0.4, 0)

    # subplot(224)
    Lx, phi, rate, nb = read(file=file4)
    rescale(axes[1][1], Lx, phi, rate, nb, 600, 100, 100, 0.4, 0.02)

    bbox = dict(edgecolor="k", fill=False)
    order = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axes.flat):
        ax.text(
            0.90,
            0.93,
            order[i],
            transform=ax.transAxes,
            bbox=bbox,
            fontsize="large")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_nb_Lx(Lx, nb, lamb, Lr):
    for i in range(Lx.size):
        plt.plot(1/Lx[i], nb[i]/Lx[i], marker=mk[nb[i]-1], color=clist[i])
    plt.show()


def plot_phi_Lx(Lx, phi, nb):
    for i in range(Lx.size):
        plt.plot(Lx[i], phi[i], marker=mk[nb[i]-1], color=clist[i])
    plt.show()
    plt.close()


def plot_rescale(Lx, nb, phi, lamb):
    Lr = Lx - nb * lamb
    l_rescaled = Lr / (nb * lamb)
    circle, = plt.plot(l_rescaled, phi, "o")
    plt.show()


if __name__ == "__main__":
    if os.path.exists("E:"):
        drive = "E:"
    else:
        drive = "D:"
    rescale_4_panel()
    mk = [
        ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "P",
        "p", "*", "h", "H", "+", "x", "X", "D", "d", "|", "-"
    ]
    mk = MarkerStyle.filled_markers
    print(len(mk))
    # file1 = r"E:\data\random_torque\bands\Lx\350_0.dat"
    # file1 = r"E:\data\random_torque\bands\Lx\old\350_20_200.dat"
    file1 = drive + r"/data/random_torque/bands/Lx/old/400_20.dat"

    lamb = 400
    Lx, phi, rate, nb = read(file=file1)
    Lr = Lx - nb * lamb
    dLr = Lr.max() - Lr.min()
    Lr0 = Lr.min()
    clist = plt.cm.jet([(i-Lr0)/dLr for i in Lr])
    # # plot_nb_Lx(Lx, nb, lamb, Lr)

    plot_phi_Lx(Lx, phi, nb)
    # plot_rescale(Lx, nb, phi, 220)
