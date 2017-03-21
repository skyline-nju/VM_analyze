# plot phi against Lr/nb*lambda

import os
import numpy as np
import matplotlib.pyplot as plt
from read_npz import sample_average


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
            except:
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


def subplot(ax, Lx, phi, rate, nb, lamb, Ly, dLx, eta, eps, rho0=1):
    Lr = Lx - nb * lamb
    l = Lr / (nb * lamb)
    ax.plot(l, phi, "o")

    z = np.polyfit(l, phi, 1)
    x = np.linspace(l.min(), l.max(), 100)
    y = x * z[0] + z[1]
    line, = ax.plot(x, y, "--")
    ax.text(
        0.05,
        0.05,
        r"$a=%f$" % z[0],
        transform=ax.transAxes,
        color=line.get_c())
    ax.text(
        0.05,
        0.13,
        r"$\Phi_c=%f$" % z[1],
        transform=ax.transAxes,
        color=line.get_c())
    ax.text(0.38, 0.82, r"$\eta=%g, \epsilon=%g, \rho_0=%g$" % (eta, eps, rho0), transform=ax.transAxes)
    ax.text(0.38, 0.75, r"$L_y=%d, \Delta L_x=%d, \lambda=%d$" % (Ly, dLx, lamb), transform=ax.transAxes)
    


if __name__ == "__main__":
    file1 = r"E:\data\random_torque\bands\Lx\350_0.dat"
    path2 = r"E:\data\random_torque\bands\Lx\snapshot\eps20"
    file3 = r"E:\data\random_torque\bands\Lx\old\400_0.dat"
    file4 = r"E:\data\random_torque\bands\Lx\old\400_20.dat"

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))

    # subplot(221)
    Lx, phi, rate, nb = read(file=file1)
    subplot(axes[0][0], Lx, phi, rate, nb, 180, 200, 20, 0.35, 0)

    # subplot(222)
    Lx, phi, rate, nb = read(path=path2, Lx=range(300, 1020, 20), eps=20)
    subplot(axes[0][1], Lx, phi, rate, nb, 220, 200, 20, 0.35, 0.02)

    # subplot(223)
    Lx, phi, rate, nb = read(file=file3)
    subplot(axes[1][0], Lx, phi, rate, nb, 400, 100, 100, 0.4, 0)

    # subplot(224)
    Lx, phi, rate, nb = read(file=file4)
    subplot(axes[1][1], Lx, phi, rate, nb, 600, 100, 100, 0.4, 0.02)

    axes[0][0].set_ylabel(r"$\Phi$")
    axes[1][0].set_ylabel(r"$\Phi$")
    axes[1][0].set_xlabel(r"$L_r/n_b\lambda$")
    axes[1][1].set_xlabel(r"$L_r/n_b\lambda$")

    bbox = dict(edgecolor="k", fill=False)
    order = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axes.flat):
        ax.text(0.92, 0.93, order[i], transform=ax.transAxes, bbox=bbox)

    plt.tight_layout(pad=0.6)
    # plt.show()
    plt.savefig(
        r"E:\report\quenched_disorder\report\fig\band_rescale.pdf", dpi=300)
    plt.close()
