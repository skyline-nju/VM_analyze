import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")
try:
    from corr2d import add_line
except ModuleNotFoundError:
    print("failed to find module add_line")
    sys.exit()


def get_para(fname):
    s = fname.split("_")
    L = int(s[1])
    eta = float(s[2])
    eps = float(s[3])
    rho0 = float(s[4])
    return L, eta, eps, rho0


def plot_Cq_3d(files):
    L, eta, eps, rho0 = get_para(files[0])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for fname in files:
        with open(fname) as f:
            lines = f.readlines()
            cv_para = np.zeros(len(lines))
            cv_perp = np.zeros_like(cv_para)
            crho_para = np.zeros_like(cv_para)
            crho_perp = np.zeros_like(cv_para)
            q = np.zeros_like(cv_para)
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                q[i] = float(s[0])
                cv_para[i] = float(s[1])
                cv_perp[i] = float(s[2])
                crho_para[i] = float(s[3])
                crho_perp[i] = float(s[4])
        ax1.plot(q, cv_para, "o", label=r"$C_v(q_{\parallel})$")
        ax1.plot(q, cv_perp, "s", label=r"$C_v (q_{\perp})$")
        ax2.plot(q, crho_para, "o", label=r"$C_\rho (q_{\parallel})$")
        ax2.plot(q, crho_perp, "s", label=r"$C_\rho (q_{\perp})$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$q/2\pi$")
    ax1.legend(fontsize="large")
    ax1.set_xlim(0.01)
    ax1.set_ylim(ymax=1000)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$q/2\pi$")
    ax2.set_xlim(0.01)
    ax2.set_ylim(ymax=2000)
    ax2.legend()

    if eps == 0.:
        add_line.add_line(ax1, 0, 0.6, 0.5, -1.65,
                          scale="log", label=r"$-1.65$")
        add_line.add_line(ax1, 0.5, 0.5, 0.8, -2.0,
                          scale="log", label=r"$-2.0$", yl=0.55)
        add_line.add_line(ax2, 0, 0.8, 0.5, -1.65,
                          scale="log", label=r"$-1.65$")
        add_line.add_line(ax2, 0.5, 0.75, 0.9, -2.55,
                          scale="log", label=r"$-2.55$", yl=0.6)
    elif eps == 0.06:
        add_line.add_line(ax1, 0.4, 0.4, 0.8, -2,
                          scale="log", label=r"$-2$")
        add_line.add_line(ax1, 0.1, 0.99, 0.6, -2.72,
                          scale="log", label=r"$-2.72$", yl=0.85)
        # add_line.add_line(ax2, 0, 0.8, 0.5, -1.65,
        #                   scale="log", label=r"$-1.65$")
        add_line.add_line(ax2, 0.2, 0.95, 0.95, -2,
                          scale="log", label=r"$-2$", yl=0.8)
    elif eps == 0.12:
        add_line.add_line(ax1, 0.2, 0.6, 0.8, -2,
                          scale="log", label=r"$-2$")
        add_line.add_line(ax1, 0.1, 0.99, 0.6, -2.7,
                          scale="log", label=r"$-2.7$", yl=0.85)
        # add_line.add_line(ax2, 0, 0.8, 0.5, -1.65,
        #                   scale="log", label=r"$-1.65$")
        add_line.add_line(ax2, 0.2, 0.95, 0.95, -2,
                          scale="log", label=r"$-2$", yl=0.8)
    elif eps == 0.18:
        add_line.add_line(ax1, 0.1, 0.7, 0.9, -2.1,
                          scale="log", label=r"$-2.1$")
        add_line.add_line(ax1, 0.1, 0.99, 0.5, -2.7,
                          scale="log", label=r"$-2.7$", yl=0.85)
        # add_line.add_line(ax2, 0, 0.8, 0.5, -1.65,
        #                   scale="log", label=r"$-1.65$")
        add_line.add_line(ax2, 0.2, 0.97, 0.95, -2,
                          scale="log", label=r"$-2$", yl=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(r"$L=%d,\ \eta=%g,\ \epsilon=%g,\ \rho_0=%g$" %
                 (L, eta, eps, rho0), fontsize="x-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d")
    f1 = "cq_240_0.20_0.060_1.0_12.dat"
    f2 = "cq_240_0.20_0.060_1.0_21.dat"
    plot_Cq_3d([f1, f2])
