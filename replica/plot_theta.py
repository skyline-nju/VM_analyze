import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def get_theta0(L, eta, seed, eps=None, disorder_t="RT"):
    if disorder_t == "RT":
        sep = "."
        if eps is None:
            pat = f"p{L}.{eta*1000:g}.*.{seed}.*.dat"
        else:
            pat = f"p{L}.{eta*1000:g}.{eps*1000:g}.{seed}.*.dat"
    elif disorder_t == "RP":
        sep = "_"
        if eps is None:
            pat = f"phi_RP_{L}_{eta:g}_*_{seed}_*.dat"
        else:
            pat = f"phi_RP_{L}_{eta:g}_{eps:g}_{seed}_*.dat"
    print(pat)
    files = glob.glob(pat)
    theta0_dict = {}
    for f in files:
        s = f.rstrip(".dat").split(sep)
        if disorder_t == "RT":
            epsilon = float(s[2]) / 1000
            theta0 = int(s[4])
        elif disorder_t == "RP":
            epsilon = float(s[4])
            theta0 = int(s[6])
        if epsilon not in theta0_dict:
            theta0_dict[epsilon] = [theta0]
        else:
            theta0_dict[epsilon].append(theta0)
    if eps is None:
        return theta0_dict
    else:
        return sorted(theta0_dict[eps])


def plot_theta_varied_eps(L, eta, seed, disorder_t="RT", wall=None):
    def set_fig(n_eps):
        if n_eps == 1:
            figsize = (4, 4)
            nrows = ncols = 1
        elif n_eps == 2:
            figsize = (6, 4)
            nrows = 1
            ncols = 2
        elif n_eps == 3:
            figsize = (8, 4)
            nrows = 1
            ncols = 3
        elif n_eps <= 6:
            figsize = (8, 8)
            nrows = 2
            ncols = 3
        else:
            figsize = (8, 12)
            nrows = 3
            ncols = 3
            if n_eps > 9:
                n_eps = 9
        return figsize, nrows, ncols, n_eps

    if disorder_t == "RT":
        dest_dir = f"E:/data/random_torque/replica2/L={L}"
        if wall == "y":
            dest_dir += "_wall_y"
    elif disorder_t == "RP":
        dest_dir = "E:/data/random_potential/replicas/serials"
    os.chdir(dest_dir)
    theta0_dict = get_theta0(L, eta, seed, disorder_t=disorder_t)
    print("eps =", sorted(theta0_dict.keys()))
    n_eps = len(theta0_dict.keys())
    figsize, nrows, ncols, n_eps = set_fig(n_eps)
    print(f"ncols={ncols}, nrwos={nrows}")
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    for i, eps in enumerate(sorted(theta0_dict.keys())[:n_eps]):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        for theta0 in sorted(theta0_dict[eps]):
            if disorder_t == "RT":
                fin = f"p{L}.{eta*1000:g}.{eps*1000:g}.{seed}.{theta0:03d}.dat"
            elif disorder_t == "RP":
                fin = f"phi_RP_{L}_{eta:g}_{eps:g}_{seed}_{theta0:03d}.dat"

            print(fin)
            phi, theta = read_phi_theta(fin, 0)
            x = (np.arange(phi.size) + 1) * 100
            theta = untangle(theta)
            ax.plot(theta, x)
            ax.set_title(f"({alphabet[i]}) $\\epsilon={eps:.3f}$")
    plt.suptitle(f"RT: $L={L}, \\eta={eta:g},$ seed={seed}",
                 fontsize="x-large")
    plt.show()
    plt.close()


def plot_phi_theta(L=256,
                   eta=0.3,
                   eps=0.01,
                   seed=30350000,
                   ncut=10000,
                   wall=None):
    dest_dir = f"E:/data/random_torque/replica2/L={L}"
    if wall == "y":
        dest_dir += "_wall_y"
    os.chdir(dest_dir)
    theta0_arr = get_theta0(L, eta, seed, eps)
    plt.figure(figsize=(6, 8), constrained_layout=True)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, projection="polar")
    for theta0 in theta0_arr:
        file_in = f"p{L}.{eta*1000:g}.{eps*1000:g}.{seed}.{theta0:03d}.dat"
        phi, theta = read_phi_theta(file_in, 0)
        x = (np.arange(phi.size) + 1) * 100
        theta = untangle(theta)
        phi_m = np.mean(phi[ncut:])
        phi_std = np.std(phi[ncut:])
        ax1.plot(x, phi, label=r"$%.4f, %.4f$" % (phi_m, phi_std))
        ax2.plot(theta, x)
    ax1.legend()
    plt.show()
    plt.close()


def plot_phi_theta3(L, eta, eps, seed, ncut, wall="none"):
    dest_dir = f"E:/data/random_torque/replica2/L={L}"
    if wall == "y":
        dest_dir += "_wall_y"
    os.chdir(dest_dir)
    theta0_arr = get_theta0(L, eta, seed, eps)
    plt.figure(figsize=(6, 9))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313, projection="polar")
    for theta0 in theta0_arr:
        file_in = f"p{L}.{eta*1000:g}.{eps*1000:g}.{seed}.{theta0:03d}.dat"
        phi, theta = read_phi_theta(file_in, 0)
        x = (np.arange(phi.size) + 1) * 100
        theta = untangle(theta)
        phi_m = np.mean(phi[ncut:])
        phi_std = np.std(phi[ncut:])
        label = f"${theta0}\\degree, {phi_m:.5f}, {phi_std:.4f}$"
        ax1.plot(x, phi, label=label)
        ax2.plot(x, theta / np.pi)
        ax3.plot(theta, x)

    ax2.axhline(1, linestyle="dashed")
    ax1.set_xlabel(r"$t$")
    ax2.set_xlabel(r"$t$")
    ax3.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\phi$")
    ax2.set_ylabel(r"$\theta/\pi$")
    ax3.set_ylabel(r"$t$")
    ax1.legend(title=r"$\theta_0,\langle\phi\rangle, \sigma_\phi=$")
    plt.suptitle(f"RS: $L={L}, \\eta={eta:g}, \\epsilon={eps:g}$")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # L = 256
    # eta = 0.05
    # seed = 30370000
    # wall = None
    # ncut = 10000
    # disorder = "RP"
    # plot_theta_varied_eps(L, eta, seed, wall=wall, disorder_t=disorder)

    L = 2048
    eta = 0.18
    eps = 0.0
    wall = "y"
    ncut = 2000
    seed = 20200713
    disorder = "RT"
    plot_phi_theta3(L, eta, eps, seed, ncut, wall)
