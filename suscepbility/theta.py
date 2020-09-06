import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def read_theta(fname, ncut=3000):
    f = open(fname, "r")
    lines = f.readlines()[ncut:]
    theta = np.array([float(i.rstrip("\n").split("\t")[1]) for i in lines])
    f.close()
    return theta


def read_phi_theta(fname, ncut=3000):
    f = open(fname, "r")
    lines = f.readlines()[ncut:]
    # phi, theta = np.zeros((2, len(lines)))
    phi, theta = [], []
    for i, line in enumerate(lines):
        try:
            s = line.rstrip("\n").split("\t")
            phi.append(float(s[0]))
            theta.append(float(s[1]))
            # phi[i], theta[i] = float(s[0]), float(s[1])
        except ValueError:
            print(fname)
    f.close()
    phi = np.array(phi)
    theta = np.array(theta)
    return phi, theta


def untangle(theta):
    theta_new = np.zeros_like(theta)
    theta_new[0] = theta[0]
    phase = 0.
    threshold = np.pi
    offset = 2 * np.pi
    for i in range(1, theta.size):
        d_theta = theta[i] - theta[i - 1]
        if d_theta > threshold:
            phase -= offset
        elif d_theta < -threshold:
            phase += offset
        theta_new[i] = theta[i] + phase
    return theta_new


def cal_theta_var(L, eps, eta=0.18, disorder_t="RF"):
    if disorder_t == "RF":
        os.chdir("E:\\data\\random_field\\normalize_new\\scaling")
        if L > 1000:
            ncut = 4000
        else:
            ncut = 3000
        pat = "phi_rf_%d_%g_%g_*.dat" % (L, eta, eps)
        files = glob.glob("serials_CSRC\\%s" % pat)
        files += glob.glob("serials_BM\\%s" % pat)
        files += glob.glob("serials_tanglou\\%s" % pat)
        files += glob.glob("serials\\%s" % pat)
        para_sep = "_"
        dest_dir = "D:\\data\\VM2d\\random_field"
    else:
        pass
    phi_mean, phi_var, theta_var = np.zeros((3, len(files)))
    seed = []
    for i, f in enumerate(files):
        phi, theta = read_phi_theta(f, ncut)
        phi_mean[i] = np.mean(phi)
        phi_var[i] = np.var(phi)
        theta_var[i] = np.var(untangle(theta))
        seed.append(f.rstrip(".dat").split(para_sep)[-1])
    if phi_mean.size > 0:
        with open("%s\\theta_var\\%d_%g_%g.dat" % (dest_dir, L, eta, eps),
                  "w") as f:
            for i in range(phi_mean.size):
                f.write("%.10f\t%.10f\t%.10f\t%s\n" %
                        (theta_var[i], phi_mean[i], phi_var[i], seed[i]))


def read_theta_var(L, eps, eta=0.18, disorder_t="RF"):
    if disorder_t == "RF":
        dest_dir = "D:\\data\\VM2d\\random_field\\theta_var"
    else:
        dest_dir = "D:\\data\\VM2d\\random_torque\\theta_var"
    with open("%s\\%d_%g_%g.dat" % (dest_dir, L, eta, eps), "r") as f:
        lines = f.readlines()
        theta_var, phi_mean, phi_var, seed = np.zeros((4, len(lines)))
        for i, line in enumerate(lines):
            s = line.rstrip("\n").split("\t")
            theta_var[i], phi_mean[i], phi_var[i], seed[i] = float(
                s[0]), float(s[1]), float(s[2]), int(s[3])
    return theta_var, phi_mean, phi_var, seed


def plot_theta_var(L, eps, eta=0.18, disorder_t="RF"):
    theta_var, phi_mean, phi_var, seed = read_theta_var(
        L, eps, eta, disorder_t)
    fig, (ax1, ax2) = plt.subplots(nrows=1,
                                   ncols=2,
                                   constrained_layout=True,
                                   sharex=True)
    ax1.plot(theta_var, phi_mean, "o")
    ax2.plot(theta_var, phi_var, "o")
    ax1.set_xscale("log")
    plt.show()
    plt.close()


def plot_theta_var_all(eps, eta=0.18, disorder_t="RF", y="phi_var"):
    L = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724, 1024, 1448])
    fig, axes = plt.subplots(nrows=3,
                             ncols=4,
                             figsize=(12, 8),
                             constrained_layout=True,
                             sharex=True,
                             sharey=True)
    for i, ax in enumerate(axes.flat):
        try:
            theta_var, phi_mean, phi_var, seed = read_theta_var(
                L[i], eps, eta, disorder_t)
            if y == "phi_var":
                ax.plot(theta_var, phi_var, "o", fillstyle="none", ms=3)
                ax.set_yscale("log")
            else:
                ax.plot(theta_var, phi_mean, "o", fillstyle="none", ms=3)

            ax.set_xscale("log")
        except FileNotFoundError:
            print("no data for L=%d, eps=%g" % (L[i], eps))
    plt.show()


def get_slope(L, phi):
    slope, L_mid = np.zeros((2, L.size - 1))
    for i in range(1, L.size):
        slope[i - 1] = np.log(phi[i - 1] / phi[i]) / np.log(L[i] / L[i - 1])
        L_mid[i - 1] = np.sqrt(L[i] * L[i - 1])
    return L_mid, slope


def get_phi_chi(eps, eta=0.18, disorder_t="RF", ret="new"):
    if eps == 0.08 or eps == 0.09:
        L_arr = np.array(
            [32, 46, 64, 90, 128, 180, 256, 362, 512, 724, 1024, 1448])
        if eps == 0.09:
            threshold = [100, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.008, 0.01, 0.02]
        elif eps == 0.08:
            threshold = [100, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.04, 0.01, 0.01]
    elif eps == 0.1:
        L_arr = np.array([32, 46, 64, 90, 128, 180, 256, 362, 512, 724])
        threshold = [1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]

    chi_con, chi_dis, phi = np.zeros((3, L_arr.size))
    chi_con2, chi_dis2, phi2 = np.zeros((3, L_arr.size))
    for i, L in enumerate(L_arr):
        theta_var, phi_mean, phi_var, seed = read_theta_var(
            L, eps, eta, disorder_t)
        chi_con[i] = np.mean(phi_var) * L**2
        chi_dis[i] = np.var(phi_mean) * L**2
        phi[i] = np.mean(phi_mean)

        mask = theta_var < threshold[i]
        phi_var2 = phi_var[mask]
        phi_mean2 = phi_mean[mask]
        chi_con2[i] = np.mean(phi_var2) * L**2
        chi_dis2[i] = np.var(phi_mean2) * L**2
        phi2[i] = np.mean(phi_mean2)
    if ret == "new":
        return L_arr, phi2, chi_con2, chi_dis2
    else:
        return L_arr, phi, chi_con, chi_dis, phi2, chi_con2, chi_dis2


def plot_scaling(eps, eta=0.18, disorder_t="RF"):
    L_arr, phi, chi_con, chi_dis, phi2, chi_con2, chi_dis2 \
        = get_phi_chi(eps, eta, disorder_t, "full")

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                        ncols=3,
                                        sharex=True,
                                        constrained_layout=True)
    ax1.plot(L_arr, chi_dis / L_arr**2, "-o")
    ax1.plot(L_arr, chi_dis2 / L_arr**2, "-s")

    ax2.plot(L_arr, chi_con / L_arr**2, "-o")
    ax2.plot(L_arr, chi_con2 / L_arr**2, "-s")

    ax3.plot(L_arr, phi, "-o")
    ax3.plot(L_arr, phi2, "-s")

    ax4 = ax3.twinx()
    Lmid, slope = get_slope(L_arr, phi)
    ax4.plot(Lmid, slope, "o")
    Lmid, slope2 = get_slope(L_arr, phi2)
    ax4.plot(Lmid, slope2, "s")

    # from cal_phi_chi import get_sample_ave_phi
    # L_arr = np.array([724, 1024, 1448])
    # phi2, chi_dis2, chi_con2 = np.zeros((3, L_arr.size))
    # for i, L in enumerate(L_arr):
    #     phi1, chi_dis1, chi_con1, phi2[i], chi_dis2[i], chi_con2[i]  \
    #         = get_sample_ave_phi(L, eps, eta, disorder_t)
    # ax2.plot(L_arr, chi_con2 / L_arr ** 2, "v")
    # ax1.plot(L_arr, chi_dis2 / L_arr ** 2, "v")
    # Lmid, slope = get_slope(L_arr, phi2)
    # ax4.plot(Lmid, slope, "v")
    ax4.set_yscale("log")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    plt.show()
    plt.close()

    for i in range(L_arr.size):
        print(L_arr[i], chi_con2[i])


if __name__ == "__main__":
    # disorder_t = "RF"
    # L = 724
    # eta = 0.18
    # eps = 0.1
    # cal_theta_var(L, eps, eta, disorder_t)
    # plot_theta_var(L, eps, eta, disorder_t)
    # plot_theta_var_all(eps, y="phi_var")
    # plot_scaling(eps)

    # L_arr = np.array([724])
    # for L in L_arr:
    #     print("L =", L)
    #     cal_theta_var(L, eps, eta, disorder_t)

    # str_in = input("input: ")
    # print(" your input is", str_in)
    # folder = "E:\\data\\random_torque\\statistic\\512 eta=0.18\\serials"
    # folder = "E:\\data\\random_torque\\statistic\\512 seed=9999\\serials"
    # f1 = "s_0.180_0.020_512_9999.dat"

    # folder = "E:\\data\\random_torque\\statistic\\512"
    # f1 = "0.050_10086.dat"

    folder = r"E:\data\random_torque\statistic\1024\serials"

    seed = 1016
    eps_arr = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    fig, (ax1, ax2) = plt.subplots(ncols=2,
                                   figsize=(8, 6),
                                   constrained_layout=True,
                                   sharex=True)
    for eps in eps_arr:
        f1 = "p_1024_0.180_%.3f_1.000_%d.dat" % (eps, seed)
        phi, theta = read_phi_theta("%s\\%s" % (folder, f1))
        theta_new = untangle(theta)
        x = np.arange(theta.size) * 100
        line, = ax1.plot(x, theta_new / np.pi, label="%g" % eps)
        if eps <= 0.04:
            ax2.plot(x, theta_new / np.pi, label="%g" % eps, c=line.get_c())
    ax1.legend(title=r"$\epsilon=$")
    ax1.set_xlabel(r"$t$", fontsize="x-large")
    ax1.set_ylabel(r"$\theta$", fontsize="x-large")
    ax2.set_xlabel(r"$t$", fontsize="x-large")
    ax2.legend(title=r"$\epsilon=$", loc="best")

    plt.suptitle(r"RS: $L=%d, \eta=%g,$ seed=%d" % (1024, 0.18, seed),
                 fontsize="x-large")
    plt.show()
    plt.close()
