"""
    Plot theta curves for each sample to obtain replica number.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
try:
    from suscepbility.theta import read_phi_theta, untangle
except ImportError:
    print("error when import add_line")


def get_serials_files(eta, eps, L, disorder_t="RT", seed=None, theta0=None):
    if disorder_t == "RT":
        folder = r"E:\data\random_torque\replica\serials"
        if seed is None:
            pat = "p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 1000)
        else:
            pat = "p%d.%g.%g.%d.*.dat" % (L, eta * 1000, eps * 1000, seed)
    elif disorder_t == "RC":
        folder = r"E:\data\random_potential\replicas\serials2"
        if seed is None:
            pat = "phi_RP_%d_%g_%g_*.dat" % (L, eta, eps)
        else:
            pat = "phi_RP_%d_%g_%g_%d_*.dat" % (L, eta, eps, seed)
    if seed is not None and theta0 is not None:
        if disorder_t == "RT":
            f = "%s/p%d.%g.%g.%d.%03d.dat" % (folder, L, eta * 1000,
                                              eps * 1000, seed, theta0)
        elif disorder_t == "RC":
            f = "%s/phi_RP_%d_%g_%g_%d_%03d.dat" % (folder, L, eta, eps, seed,
                                                    theta0)
        return f
    else:
        files = glob.glob(r"%s\%s" % (folder, pat))
        return files


def get_seeds(eta, eps, L, disorder_t="RT"):
    files = get_serials_files(eta, eps, L, disorder_t)
    seeds = []
    for file in files:
        if disorder_t == "RT":
            seed = int(os.path.basename(file).split(".")[3])
        elif disorder_t == "RC":
            seed = int(os.path.basename(file).split("_")[5])
        if seed not in seeds:
            seeds.append(seed)
    print("find %d samples" % len(seeds))
    return sorted(seeds)


def get_theta0(fname, disorder_t):
    if disorder_t == "RT":
        theta0 = int(os.path.basename(fname).split(".")[4])
    elif disorder_t == "RC":
        theta0 = int(os.path.basename(fname).rstrip(".dat").split("_")[6])
    return theta0


def get_mean_var(eta, eps, L, seed, disorder_t="RT", ncut=15000):
    if disorder_t == "RT":
        folder = r"E:\data\random_torque\replica\serials"
    else:
        pass
    pat = "p%d.%g.%g.%d.*.dat" % (L, eta * 1000, eps * 1000, seed)
    files = glob.glob("%s\\%s" % (folder, pat))
    mean_arr = []
    var_arr = []
    for file in files:
        phi, theta = read_phi_theta(file, ncut)
        mean_arr.append(np.mean(phi))
        var_arr.append(np.var(phi))
    return mean_arr, var_arr


def plot_replicas(eta, eps, L, seed, disorder_t="RT", ncut=15000):
    files = get_serials_files(eta, eps, L, disorder_t, seed)
    theta0 = [get_theta0(f, disorder_t) for f in files]
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211, polar=False)
    ax2 = fig.add_subplot(212, polar=True)
    mean_arr = []
    var_arr = []
    for theta_i in sorted(theta0):
        file = get_serials_files(eta, eps, L, disorder_t, seed, theta_i)
        beg = 0
        phi, theta = read_phi_theta(file, beg)
        x = (np.arange(phi.size) + beg + 1) * 100
        ax1.plot(x, theta / np.pi, label="%g" % theta_i)
        theta = untangle(theta)
        ax2.plot(theta, x)
        phi_mean, phi_var = np.mean(phi[ncut:]), np.var(phi[ncut:])
        mean_arr.append(phi_mean)
        var_arr.append(phi_var)
        # ax3.plot(x, phi, label="%.4f, %.5f" % (phi_mean, phi_var))
    # ax3.set_xlabel(r"$t$")
    # ax3.set_ylabel(r"$m$")
    ax2.set_ylabel(r"$\theta/\pi$")
    ax1.set_ylabel(r"$\theta/\pi$")
    ax1.legend(title=r"$\theta_0=$")
    # ax3.legend(title=r"$\overline{m}, \langle (m-\overline{m})^2\rangle =$")
    title = r"$L=%d, \eta=%g, \epsilon=%g,$ seed=%d" % (L, eta, eps, seed)
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
    return mean_arr, var_arr


def count_replicas(eta, eps, L, disorder_t="RT"):
    seeds_old = []
    if disorder_t == "RT":
        outfile = r"D:\data\VM2d\random_torque\replica\%d_%g_%g.dat" % (L, eta,
                                                                        eps)
    elif disorder_t == "RC":
        outfile = r"D:\data\VM2d\random_potential\replica\%d_%g_%g.dat" % (
            L, eta, eps)
    try:
        f = open(outfile, "r")
        lines = f.readlines()
        for line in lines:
            seeds_old.append(int(line.split("\t")[0]))
        f.close()
    except FileNotFoundError:
        pass
    seeds = get_seeds(eta, eps, L, disorder_t)
    if os.path.exists(outfile):
        f = open(outfile, "a")
    else:
        f = open(outfile, "w")
    for i, seed in enumerate(seeds):
        if seed in seeds_old:
            continue
        mean_arr, var_arr = plot_replicas(eta, eps, L, seed, disorder_t)
        str_in = input("%d/%d: please record replica number: " %
                       (i, len(seeds)))
        if str_in[0] == "q":
            break
        elif str_in[0].isdigit():
            line = "%d\t%s" % (seed, str_in[0])
            if len(str_in) == 2:
                ncut = int(str_in[1]) * 1000
                mean_arr, var_arr = get_mean_var(eta, eps, L, seed, "RT", ncut)
            for phi in mean_arr:
                line += "\t%.8f" % phi
            for var in var_arr:
                line += "\t%.8f" % var
            line += "\n"
            f.write(line)
    f.close()


def plot_replica_num(disorder="RT"):
    if disorder == "RT":
        eta = 0.18
        eps = 0.035
        os.chdir("D:/data/VM2d/random_torque/replica")
        L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
    elif disorder == "RC":
        eta = 0
        eps = 0.2
        os.chdir("D:/data/VM2d/random_potential/replica")
        L = np.array([32, 64, 128, 256, 512])
    n = np.zeros(L.size)
    fraction = np.zeros((5, L.size))
    for i in range(L.size):
        fin = "%d_%g_%g.dat" % (L[i], eta, eps)
        count = 0
        with open(fin, "r") as f:
            lines = f.readlines()
            for line in lines:
                k = int(line.split("\t")[1])
                if k > 0:
                    count += 1
                    n[i] += k
                fraction[k][i] += 1
            n[i] /= count
            for k in range(5):
                fraction[k][i] /= len(lines)
    fig, (ax1, ax2) = plt.subplots(ncols=2,
                                   figsize=(8, 4),
                                   constrained_layout=True)
    ax1.plot(L, n, "o")
    ax1.set_ylabel(r"$\langle n\rangle$")
    ax1.set_xlabel(r"$L$")
    for k in range(5):
        if k == 0:
            label = "nonsteady"
            for j in range(L.size):
                print(L[j], fraction[k][j])
        else:
            label = r"$n=%d$" % k
        mask = fraction[k] > 0
        ax2.loglog(L[mask], fraction[k][mask], "-o", fillstyle="none", label=label)
    ax2.legend()
    ax2.set_xlabel(r"$L$")
    ax2.set_ylabel("Probability")
    plt.show()
    plt.close()


def plot_suscept():
    os.chdir("D:\\data\\VM2d\\random_torque\\replica")
    L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
    chi = np.zeros(L.size)
    for i in range(L.size):
        fin = "%d_%g_%g.dat" % (L[i], 0.18, 0.035)
        count = 0
        with open(fin, "r") as f:
            lines = f.readlines()
            for line in lines:
                k = int(line.split("\t")[1])
                if k == 2:
                    s = line.rstrip("\n").split("\t")
                    phi_arr = []
                    if len(s) == 10:
                        n = 6
                    elif len(s) == 18:
                        n = 10
                    for j in range(2, n):
                        phi_arr.append(float(s[j]))
                    phi_arr = np.array(phi_arr)
                    phi_arr_new = np.array([phi_arr.max(), phi_arr.min()])
                    chi[i] += np.var(phi_arr_new)
                    count += 1
        chi[i] /= count
        print(L[i], chi[i])


def plot_suscept2():
    os.chdir("D:\\data\\VM2d\\random_torque\\replica")
    L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
    chi = np.zeros(L.size)
    for i in range(L.size):
        fin = "%d_%g_%g.dat" % (L[i], 0.18, 0.035)
        count = 0
        phi_mean = []
        with open(fin, "r") as f:
            lines = f.readlines()
            for line in lines:
                s = line.rstrip("\n").split("\t")
                # phi_arr = []
                if len(s) == 10:
                    n = 6
                elif len(s) == 18:
                    n = 10
                for j in range(2, n):
                    phi_mean.append(float(s[j]))
                # phi_arr = np.array(phi_arr)
                # phi_mean.append(phi_arr.max())
        # chi[i] /= count
        phi_mean = np.array(phi_mean)
        chi[i] = np.var(phi_mean)
        print(L[i], chi[i])


def plot_suscept3(disorder="RT"):
    if disorder == "RT":
        eta = 0.18
        eps = 0.035
        os.chdir("D:/data/VM2d/random_torque/replica")
        L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
    elif disorder == "RC":
        eta = 0
        eps = 0.2
        os.chdir("D:/data/VM2d/random_potential/replica")
        L = np.array([32, 64, 128, 256, 512])
    chi = np.zeros(L.size)

    for i in range(L.size):
        fin = "%d_%g_%g.dat" % (L[i], eta, eps)
        phi_arr = []
        with open(fin, "r") as f:
            lines = f.readlines()
            for line in lines:
                s = line.rstrip("\n").split("\t")
                if len(s) == 10:
                    n = 6
                elif len(s) == 18:
                    n = 10
                # j = np.random.randint(2, n, size=1)[0]
                phi_arr.append(np.mean(np.array([float(s[j]) for j in range(2, n)])))
                # for j in range(2, n):
                #     phi_arr.append(float(s[j]))
        # count = 0
        # phi_arr = np.array(phi_arr)
        # for j in range(phi_arr.size // 2):
        #     chi[i] += np.var(phi_arr[j*2: (j+1)*2])
        #     count += 1
        # chi[i] /= count
        phi_arr = np.array(phi_arr)
        chi[i] = np.var(phi_arr)
        print(L[i], chi[i])


def plot_Fig2gh():
    fig, (ax1, ax2) = plt.subplots(1,
                                   2,
                                   constrained_layout=True,
                                   figsize=(8, 4))

    os.chdir("D:\\data\\VM2d\\random_torque\\replica")
    L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
    n = np.zeros(L.size)
    fraction = np.zeros((5, L.size))
    for i in range(L.size):
        fin = "%d_%g_%g.dat" % (L[i], 0.18, 0.035)
        count = 0
        with open(fin, "r") as f:
            lines = f.readlines()
            for line in lines:
                k = int(line.split("\t")[1])
                if k > 0:
                    count += 1
                    n[i] += k
                fraction[k][i] += 1
            n[i] /= count
            for k in range(5):
                fraction[k][i] /= len(lines)

    for k in range(5):
        if k == 0:
            label = "nonsteady"
        else:
            label = r"$n=%d$" % k
        ax2.plot(L, fraction[k], "-o", fillstyle="none", label=label)
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$L$")
    ax2.set_ylabel("Fraction")
    ax1.set_xlabel(r"$L$")
    ax1.set_ylabel("Fraction")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # eta = 0.18
    # eps = 0.035
    # L = 32
    # plot_replica_num("RC")
    # count_replicas(eta, eps, L)
    # plot_suscept3("RT")

    L = 90
    eta = 0
    eps = 0.2
    count_replicas(eta, eps, L, "RC")

    # plot_Fig2gh()
