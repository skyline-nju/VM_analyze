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


def get_seeds(eta, eps, L, disorder_t="RT"):
    if disorder_t == "RT":
        folder = r"E:\data\random_torque\replica\serials"
    else:
        pass
    pat = "p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 1000)
    files = glob.glob("%s\\%s" % (folder, pat))
    seeds = []
    for file in files:
        seed = int(os.path.basename(file).split(".")[3])
        if seed not in seeds:
            seeds.append(seed)
    print("find %d samples" % len(seeds))
    return sorted(seeds)


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
    if disorder_t == "RT":
        folder = r"E:\data\random_torque\replica\serials"
    else:
        pass
    pat = "p%d.%g.%g.%d.*.dat" % (L, eta * 1000, eps * 1000, seed)
    files = glob.glob("%s\\%s" % (folder, pat))
    theta0 = []
    for file in files:
        theta0.append(int(os.path.basename(file).split(".")[4]))
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211, polar=False)
    ax2 = fig.add_subplot(212, polar=True)
    mean_arr = []
    var_arr = []
    for theta_i in sorted(theta0):
        file = "%s\\p%d.%g.%g.%d.%03d.dat" % (folder, L, eta * 1000,
                                              eps * 1000, seed, theta_i)
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
        try:
            f = open(outfile, "r")
            lines = f.readlines()
            for line in lines:
                seeds_old.append(int(line.split("\t")[0]))
            f.close()
        except FileNotFoundError:
            pass
    seeds = get_seeds(eta, eps, L)
    if os.path.exists(outfile):
        f = open(outfile, "a")
    else:
        f = open(outfile, "w")
    for i, seed in enumerate(seeds):
        if seed in seeds_old:
            continue
        mean_arr, var_arr = plot_replicas(eta, eps, L, seed, disorder_t)
        str_in = input(
            "%d/%d: please record replica number: " % (i, len(seeds)))
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


def plot_replica_num():
    os.chdir("D:\\data\\VM2d\\random_torque\\replica")
    L = np.array([64, 90, 128, 180, 256, 362, 512])
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
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(8, 4), constrained_layout=True)
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
        ax2.loglog(L, fraction[k], "-o", fillstyle="none", label=label)
    ax2.legend()
    ax2.set_xlabel(r"$L$")
    ax2.set_ylabel("Probability")
    plt.show()
    plt.close()


def plot_suscept():
    os.chdir("D:\\data\\VM2d\\random_torque\\replica")
    L = np.array([180, 256, 362, 512])
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
                    phi_arr = np.array(
                        [float(s[2]),
                         float(s[3]),
                         float(s[4]),
                         float(s[5])])
                    phi_arr_new = np.array([phi_arr.max(), phi_arr.min()])
                    chi[i] += np.var(phi_arr_new)
                    count += 1
        chi[i] /= count
        print(L[i], chi[i])


if __name__ == "__main__":
    # eta = 0.18
    # eps = 0.035
    # L = 64
    # count_replicas(eta, eps, L)
    plot_replica_num()
