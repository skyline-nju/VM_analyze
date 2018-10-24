import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
try:
    from corr2d.add_line import add_line
except ModuleNotFoundError:
    print("failed to find module add_line")
    sys.exit()


def cal_phi(infile, n_cut, only_phi=True, d=3):
    with open(infile) as f:
        lines = f.readlines()[n_cut:]
        # phi = np.array([float(line.split("\t")[0]) for line in lines])
        phi = []
        if d == 3:
            pos = 0
        else:
            pos = 1
        for i, line in enumerate(lines):
            try:
                phi.append(float(line.split("\t")[pos]))
            except ValueError:
                pass
        phi = np.array(phi)
        if only_phi:
            return np.mean(phi)
        else:
            if d == 3:
                L = float(infile.split("_")[1])
            else:
                L = float(infile.split("_")[4])
            chi = np.var(phi) * L ** 2
            Binder = 1 - np.mean(phi ** 4) / np.mean(phi ** 2) ** 2 / 3
            return np.mean(phi), chi, Binder


def varied_domain_size(eps, seed):
    L = [16, 22, 32, 46, 64, 80, 96]
    n_cut = 2500
    for l in L:
        filename = "phi_%d_0.20_%.3f_1.0_%d.dat" % (l, eps, seed)
        if os.path.exists(filename):
            print(l, cal_phi(filename, n_cut))


def sample_ave_phi(eps, eta=0.2, n_cut=3500):
    L = [16, 22, 32, 46, 64, 80, 96, 120]
    phi = {l: [] for l in L}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5))
    for l in L:
        files = glob.glob("phi_%d_%.2f_%.3f_1.0_*.dat" % (l, eta, eps))
        for f in files:
            phi[l].append(cal_phi(f, n_cut))
    phi_m = np.zeros(len(L))
    phi_std = np.zeros(len(L))
    for i, l in enumerate(sorted(phi.keys())):
        phi[l] = np.array(phi[l])
        n = phi[l].size
        plt.plot(np.ones(n) * l, phi[l], "o", c="#7f7f7f", alpha=0.5)
        phi_m[i] = np.mean(phi[l])
        phi_std[i] = np.std(phi[l])
        print(l, phi_m[i])
    plt.errorbar(L, phi_m, phi_std, color="r")
    plt.plot(L, phi_m, "rs")
    if eps == 0.06:
        add_line(ax, 0.6, 0.3, 1, slope=-0.004,
                 label=r"$-0.004$", scale="log", c="b")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$L$", fontsize="xx-large")
    plt.ylabel(r"$\Phi$", fontsize="xx-large")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\eta=%g, \epsilon=%g, \rho_0=1.0, d=3$" %
                 (eta, eps), fontsize="xx-large")
    plt.show()
    plt.close()


def sample_ave_phi_all(eta=0.2, n_cut=3500):
    eps = [0, 0.02, 0.06, 0.12]
    L = [16, 22, 32, 46, 64, 80, 96, 120]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    for epsilon in eps:
        phi = {l: [] for l in L}
        for l in L:
            files = glob.glob("phi_%d_%.2f_%.3f_1.0_*.dat" % (l, eta, epsilon))
            for f in files:
                phi[l].append(cal_phi(f, n_cut))
        phi_m = np.zeros(len(L))
        phi_std = np.zeros(len(L))
        for i, l in enumerate(sorted(phi.keys())):
            phi[l] = np.array(phi[l])
            phi_m[i] = np.mean(phi[l])
            phi_std[i] = np.std(phi[l])
        line, = plt.plot(L, phi_m, "o", label=r"$\epsilon=%g$" % epsilon)
        plt.errorbar(L, phi_m, phi_std, c=line.get_color())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$L$", fontsize="x-large")
    plt.ylabel(r"$\Phi$", fontsize="x-large")
    plt.title(r"$\eta=%g, \rho_0=1.0, d=3$" % eta, fontsize="xx-large")
    plt.legend(fontsize="x-large")
    plt.tight_layout()
    plt.show()
    plt.close()


def varied_eps(eta=0.2, seed=11, d=3):
    if d == 3:
        L = [32, 64, 80, 80, 96]
        n_cut = 2500
    else:
        L = [1024]
        n_cut = 4500
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    flag_80 = False
    for li in L:
        phi = {}
        chi = {}
        Binder = {}
        if d == 3:
            if li == 96:
                seed = 13
                n_cut = 3000
            elif li == 80:
                if flag_80:
                    seed = 13
                else:
                    flag_80 = True
                    seed = 11
            else:
                seed = 11
                n_cut = 2500
            files = glob.glob("phi_%d_%.2f_*_1.0_%d.dat" % (li, eta, seed))
        else:
            os.chdir(r"D:\data\random_torque\phase diagram\L=1024\order_para")
            files = glob.glob("p_%g_*_%g_%d_%d_%d.dat" %
                              (eta, 1.0, li, li, seed))
        for f in files:
            print(f)
            if d == 3:
                eps = float(f.split("_")[3])
            else:
                if li == 1024:
                    eps = float(f.split("_")[2])
            phi[eps], chi[eps], Binder[eps] = cal_phi(f, n_cut, False, d=d)
        eps_arr = np.array([i for i in sorted(phi.keys())])
        phi_arr = np.array([phi[i] for i in eps_arr])
        chi_arr = np.array([chi[i] for i in eps_arr])
        Binder_arr = np.array([Binder[i] for i in eps_arr])
        ax1.plot(eps_arr, phi_arr, "-o", label=r"$L=%d$" % li)
        ax2.plot(eps_arr, chi_arr, "-o", label=r"$L=%d$" % li)
        ax3.plot(eps_arr, Binder_arr, "-o", label=r"$L=%d$" % li)
    ax2.set_yscale("log")
    ax1.set_xlabel(r"$\epsilon$", fontsize="x-large")
    ax2.set_xlabel(r"$\epsilon$", fontsize="x-large")
    ax3.set_xlabel(r"$\epsilon$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle \phi \rangle$", fontsize="x-large")
    ax2.set_ylabel(
        r"$L^2\left [\langle \phi^2\rangle - \langle \phi\rangle^2\right]$",
        fontsize="x-large")
    ax3.set_ylabel(
        r"$1-\langle\phi^4\rangle/3 \langle\phi^2\rangle^2$",
        fontsize="x-large")

    ax1.legend(fontsize="x-large")
    ax2.legend(fontsize="x-large")
    ax3.legend(fontsize="x-large")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(r"$\eta=%.2f, \rho_0=1.0, d=%d$" %
                 (eta, d), fontsize="xx-large")
    plt.show()
    plt.close()


def one_sample(L, seed, rho0=1.0):
    files = glob.glob("phi_%d_*_%.1f_%d.dat" % (L, rho0, seed))
    for i in files:
        s = i.split("_")
        eta = float(s[2])
        eps = float(s[3])
        try:
            print(eta, eps, cal_phi(i, 2000))
        except ValueError:
            print(i)


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d")
    # varied_eps()
    # varied_eps(0.15, 111111, 2)
    # varied_domain_size()
    # one_sample(64, 11)
    # varied_domain_size(0.02, 80)
    sample_ave_phi_all()
