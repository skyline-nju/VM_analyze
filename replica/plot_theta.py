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


def get_theta0(Lx, eta, seed, eps=None, disorder_t="RT", Ly=None):
    if disorder_t == "RT":
        sep = "."
        if Ly is None:
            if eps is None:
                pat = f"p{Lx}.{eta*1000:g}.*.{seed}.*.dat"
            else:
                pat = f"p{Lx}.{eta*1000:g}.{eps*1000:g}.{seed}.*.dat"
        else:
            if eps is None:
                pat = f"p{Lx}.{Ly}.{eta*1000:g}.*.{seed}.*.dat"
            else:
                pat = f"p{Lx}.{Ly}.{eta*1000:g}.{eps*1000:g}.{seed}.*.dat"
    elif disorder_t == "RP":
        sep = "_"
        if eps is None:
            pat = f"phi_RP_{Lx}_{eta:g}_*_{seed}_*.dat"
        else:
            pat = f"phi_RP_{Lx}_{eta:g}_{eps:g}_{seed}_*.dat"
    elif disorder_t == "RF":
        sep = "_"
        if eps is None:
            pat = f"phi_rf_{Lx}_{eta:.3f}_*_{seed}_*.dat"
        else:
            pat = f"phi_rf_{Lx}_{eta:.3f}_{eps:.3f}_{seed}_*.dat"
    print(pat)
    files = glob.glob(pat)
    theta0_dict = {}
    for f in files:
        s = f.rstrip(".dat").split(sep)
        if disorder_t == "RT":
            if Ly is None:
                epsilon = float(s[2]) / 1000
                theta0 = int(s[4])
            else:
                epsilon = float(s[3]) / 1000
                theta0 = int(s[5])
        elif disorder_t == "RP" or disorder_t == "RF":
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


def get_filename(L, eta, eps, seed, theta0, disorder, Ly=None):
    if disorder == "RT":
        if Ly is None:
            fin = f"p{L}.{eta*1000:g}.{eps*1000:g}.{seed}.{theta0:03d}.dat"
        else:
            fin = f"p{L}.{Ly}.{eta*1000:g}.{eps*1000:g}." +\
                    f"{seed}.{theta0:03d}.dat"
    elif disorder == "RP":
        fin = f"phi_RP_{L}_{eta:g}_{eps:g}_{seed}_{theta0:03d}.dat"
    elif disorder == "RF":
        fin = f"phi_rf_{L}_{eta:.3f}_{eps:.3f}_{seed}_{theta0:03d}.dat"
    return fin


def get_dest_dir(disorder_t, Lx=None, Ly=None, wall=None):
    if disorder_t == "RT":
        if Ly is None:
            dest_dir = f"E:/data/random_torque/replica2/L={Lx}"
        else:
            dest_dir = "E:/data/random_torque/replica2/Rect"
    elif disorder_t == "RP":
        dest_dir = "E:/data/random_potential/replicas/serials"
    elif disorder_t == "RF":
        dest_dir = "E:/data/random_field/normalize_new/replica"
    if wall == "y":
        dest_dir += "_wall_y"
    return dest_dir


def plot_theta_varied_eps(L, eta, seed, disorder_t="RT", wall=None, start=0):
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

    dest_dir = get_dest_dir(disorder_t, L, wall=wall)
    os.chdir(dest_dir)
    theta0_dict = get_theta0(L, eta, seed, disorder_t=disorder_t)
    print("eps =", sorted(theta0_dict.keys()))
    n_eps = len(theta0_dict.keys())
    figsize, nrows, ncols, n_eps = set_fig(n_eps)
    print(f"ncols={ncols}, nrwos={nrows}")
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    for i, eps in enumerate(sorted(theta0_dict.keys())[start:n_eps + start]):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        for theta0 in sorted(theta0_dict[eps]):
            fin = get_filename(L, eta, eps, seed, theta0, disorder_t)

            print(fin)
            phi, theta = read_phi_theta(fin, 0)
            x = (np.arange(phi.size) + 1) * 100
            theta = untangle(theta)
            ax.plot(theta, x)
            ax.set_title(f"({alphabet[i]}) $\\epsilon={eps:.3f}$")
    plt.suptitle(f"{disorder_t}: $L={L}, \\eta={eta:g},$ seed={seed}",
                 fontsize="x-large")
    plt.show()
    plt.close()


def plot_phi_theta(L=256,
                   eta=0.3,
                   eps=0.01,
                   seed=30350000,
                   ncut=10000,
                   wall=None,
                   disorder_t="RT"):
    dest_dir = get_dest_dir(disorder_t, L, wall=wall)
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


def plot_phi_theta3(Lx,
                    eta,
                    eps,
                    seed,
                    ncut,
                    wall="none",
                    Ly=None,
                    disorder="RT"):
    dest_dir = get_dest_dir(disorder, Lx, Ly, wall)
    os.chdir(dest_dir)
    theta0_arr = get_theta0(Lx, eta, seed, eps, disorder, Ly)
    plt.figure(figsize=(6, 9))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313, projection="polar")
    for theta0 in theta0_arr:
        fin = get_filename(Lx, eta, eps, seed, theta0, disorder, Ly)
        phi, theta = read_phi_theta(fin, 0)
        x = (np.arange(phi.size) + 1) * 100
        theta = untangle(theta)
        phi_m = np.mean(phi[ncut:])
        phi_std = np.std(phi[ncut:])
        label = f"${theta0}\\degree, {phi_m:.5f}, {phi_std:.4f}$"
        ax1.plot(x, phi, label=label)
        ax2.plot(x, theta / np.pi)
        ax3.plot(theta, x)

    # ax2.axhline(1, linestyle="dashed")
    ax1.set_xlabel(r"$t$")
    ax2.set_xlabel(r"$t$")
    ax3.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\phi$")
    ax2.set_ylabel(r"$\theta/\pi$")
    ax3.set_ylabel(r"$t$")
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    ax1.legend(title=r"$\theta_0,\langle\phi\rangle, \sigma_\phi=$")
    if Ly is None:
        title = f"RS: $L={Lx}, \\eta={eta:g}, \\epsilon={eps:g}$"
    else:
        title = f"RS: $L_x={Lx},L_y={Ly}, \\eta={eta:g}, \\epsilon={eps:g}$"
    if disorder == "RP":
        title = title.replace("RS", "RC")
    elif disorder == "RF":
        title = title.replace("RS", "RF")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # L = 1024
    # eta = 0.
    # seed = 30370000
    # wall = None
    # ncut = 10000
    # disorder = "RP"
    # plot_theta_varied_eps(L, eta, seed, disorder, wall, start=0)

    L = 4096
    eta = 0.18
    eps = 0.035
    wall = None
    ncut = 2000
    seed = 20200712
    disorder = "RT"
    plot_phi_theta3(L, eta, eps, seed, ncut, wall)

    # Lx = 16384
    # Ly = 1024
    # eta = 0.18
    # eps = 0.035
    # wall = "y"
    # seed = 20200712
    # disorder = "RT"
    # plot_phi_theta3(Lx, eta, eps, seed, 1000, wall, Ly)

    # L = 2048
    # eta = 0.18
    # eps = 0.09
    # wall = None
    # ncut = 4000
    # seed = 20200712
    # disorder = "RF"
    # plot_phi_theta3(L, eta, eps, seed, ncut, wall, disorder=disorder)

    # L = 1024
    # eta = 0
    # eps = 0.2
    # wall = None
    # ncut = 3000
    # seed = 30370000
    # disorder = "RP"
    # plot_phi_theta3(L, eta, eps, seed, ncut, wall, disorder=disorder)
