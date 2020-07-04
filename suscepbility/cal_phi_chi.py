import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from theta import read_phi_theta, untangle


def plot_serials(L,
                 eps,
                 eta,
                 seed,
                 disorder_t,
                 phi_arr,
                 theta_arr,
                 ncut0,
                 out_dir=None,
                 beg=None,
                 end=None):
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, constrained_layout=True)
    phi_mean = np.mean(phi_arr[ncut0:])
    phi_var = np.var(phi_arr[ncut0:])
    theta = untangle(theta_arr)
    theta_mean = np.mean(theta[ncut0:])
    theta_var = np.var(theta[ncut0:])

    x = np.arange(phi_arr.size)
    axes[0].plot(x, phi_arr, label="mean=%f, var=%.3e" % (phi_mean, phi_var))
    axes[1].plot(
        x,
        theta / np.pi,
        label="mean=%.3f, var=%.3e" % (theta_mean, theta_var))
    axes[0].axvline(ncut0, c="tab:red")
    axes[1].axvline(ncut0, c="tab:red")
    axes[0].set_ylabel(r"$m$")
    axes[1].set_ylabel(r"$\theta/\pi$")
    plt.suptitle(r"%s: $L=%d, \eta=%g, \epsilon=%g, {\rm seed}=%s$" %
                 (disorder_t, L, eta, eps, seed))
    axes[0].grid(axis="x", which="both")
    axes[1].grid(axis="x", which="both")
    if out_dir is None:
        axes[0].legend()
        axes[1].legend()
        plt.draw()
        plt.waitforbuttonpress(0)
    else:
        if beg is not None and end is not None:
            phi_mean = np.mean(phi_arr[beg:end])
            phi_var = np.var(phi_arr[beg:end])
            theta_mean = np.mean(theta[beg:end])
            theta_var = np.var(theta[beg:end])
            axes[0].axvspan(
                beg,
                end,
                alpha=0.4,
                label="mean=%f, var=%.3e" % (phi_mean, phi_var))
            axes[1].axvspan(
                beg,
                end,
                alpha=0.4,
                label="mean=%.3f, var=%.3e" % (theta_mean, theta_var))
        axes[0].legend()
        axes[1].legend()
        plt.savefig("%s\\%g_%g_%d_%s.png" % (out_dir, eta, eps, L, seed))
    plt.close()
    return phi_mean, phi_var, theta_mean, theta_var


def cal_phi_chi(L, eps, eta=0.18, disorder_t="RF"):
    if disorder_t == "RF":
        os.chdir("E:\\data\\random_field\\normalize_new\\scaling")
        pat = "phi_rf_%d_%g_%g_*.dat" % (L, eta, eps)
        files = glob.glob("serials_CSRC\\%s" % pat)
        files += glob.glob("serials_BM\\%s" % pat)
        files += glob.glob("serials_tanglou\\%s" % pat)
        files += glob.glob("serials\\%s" % pat)
        files += glob.glob("tmp\\%s" % pat)
        para_sep = "_"
        dest_dir = "D:\\data\\VM2d\\random_field\\phi_chi"
    else:
        os.chdir("E:\\data\\random_torque\\susceptibility\\phi")
        pat = "p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 10000)
        files = glob.glob("eta=%.2f\\%s" % (eta, pat))
        files += glob.glob("eta=%.2f_BM\\%s" % (eta, pat))
        files += glob.glob("eta=%.2f_CSRC\\%s" % (eta, pat))
        if L <= 724:
            os.chdir("E:\\data\\random_torque\\Phi_vs_L\\eta=%.2f\\%.3f" %
                     (eta, eps))
            files += glob.glob(pat)
            if eps == 0.03 or eps == 0.02:
                pat2 = "p%d.%g.%g.*.dat" % (L, eta * 1000, eps * 1000)
                files += glob.glob(pat2)
        para_sep = "."
        dest_dir = "D:\\data\\VM2d\\random_torque\\phi_chi"

    dest_file = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L)
    if os.path.exists(dest_file):
        with open(dest_file, "r") as f:
            lines = f.readlines()
            seed_list = [int(i.split("\t")[1]) for i in lines]
        fout = open(dest_file, "a")
    else:
        seed_list = []
        fout = open(dest_file, "w")
    if L > 1000:
        ncut0 = 4000
    else:
        ncut0 = 3000
    fig_dir = "%s\\fig" % (dest_dir)
    for i, fin in enumerate(sorted(files)):
        seed = int(fin.rstrip(".dat").split(para_sep)[-1])
        if seed not in seed_list:
            phi, theta = read_phi_theta(fin, 0)
            phi_var = np.var(phi[ncut0:])
            if phi_var <= 0:
                line = "%d\t%s\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
                    0, seed, np.mean(phi[ncut0:]), phi_var,
                    np.mean(theta[ncut0:]), np.var(theta[ncut0:]))
            else:
                phi_mean, phi_var, theta_mean, theta_var = plot_serials(
                    L, eps, eta, seed, disorder_t, phi, theta, ncut0)
                str_in = input(
                    "%d/%d: please set beg, end: " % (i, len(files)))
                if str_in == "":
                    beg, end = ncut0, phi.size
                    state = 0
                else:
                    str_list = str_in.split(" ")
                    if len(str_list) == 1:
                        if str_list[0] == "n":
                            state = 2
                            beg, end = ncut0, phi.size
                        elif str_list[0] == "q":
                            break
                        else:
                            beg = int(str_list[0]) * 1000
                            end = phi.size
                            state = 1
                    elif len(str_list) == 2:
                        beg = int(str_list[0]) * 1000
                        if str_list[-1] == "n":
                            state = 2
                            end = phi.size
                        else:
                            end = int(str_list[1]) * 1000
                            state = 1
                    elif len(str_list) == 3:
                        beg = int(str_list[0]) * 1000
                        end = int(str_list[1]) * 1000
                        state = 2
                line = "%d\t%s\t%.8f\t%.8f\t%.8f\t%.8f" % (
                    state, seed, phi_mean, phi_var, theta_mean, theta_var)
                if beg == ncut0 and end == phi.size:
                    line += "\n"
                else:
                    phi_mean2, phi_var2, theta_mean2, theta_var2 = plot_serials(
                        L, eps, eta, seed, disorder_t, phi, theta, ncut0,
                        fig_dir, beg, end)
                    line += "\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
                        phi_mean2, phi_var2, theta_mean2, theta_var2)
            fout.write(line)
            if i % 10 == 0:
                fout.flush()
                os.fsync(fout)
    fout.close()


def read_data(fin):
    """
        phi_mean, phi_var, theta_mean, theta_var,
        phi_mean2, phi_var2, theta_mean2, theta_var2
    """
    with open(fin, "r") as f:
        lines = f.readlines()
        n = len(lines)
        data = np.zeros((n, 8))
        for i, line in enumerate(lines):
            A = np.array([float(i) for i in line.rstrip("\n").split("\t")[2:]])
            if A.size == 8:
                data[i] = A
            elif A.size == 6:
                data[i] = np.array([A[0], A[1], 0, A[2], A[3], A[4], 0, A[5]])
            elif A.size == 4:
                data[i, :4] = A
                data[i, 4:] = A
            elif A.size == 3:
                data[i, :4] = np.array([A[0], A[1], 0, A[2]])
                data[i, 4:] = data[i, :4]
    return data.T


def get_chi_M(eps, eta=0.18, disorder_t="RF"):
    if disorder_t == "RF":
        dest_dir = "D:\\data\\VM2d\\random_field\\phi_chi"
    else:
        dest_dir = "D:\\data\\VM2d\\random_torque\\phi_chi"
    files = glob.glob("%s\\%g_%g_*.dat" % (dest_dir, eta, eps))
    L = []
    for fin in files:
        L.append(int(os.path.basename(fin).rstrip(".dat").split("_")[2]))
    L = np.array(sorted(L))
    chi_dis, chi_con, M = np.zeros((3, L.size))
    for i in range(L.size):
        fin = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L[i])
        phi_mean, phi_var, theta_mean, theta_var,\
            phi_mean2, phi_var2, theta_mean2, theta_var2 = read_data(fin)
        chi_dis[i] = np.var(phi_mean2) * L[i] ** 2
        chi_con[i] = np.mean(phi_var2) * L[i] ** 2
        M[i] = np.mean(phi_mean2)
    return L, chi_dis, chi_con, M


def get_nonsteady_rates(eps, eta=0.18, disorder_t="RT"):
    if disorder_t == "RF":
        dest_dir = "D:\\data\\VM2d\\random_field\\phi_chi"
    else:
        dest_dir = "D:\\data\\VM2d\\random_torque\\phi_chi"
    files = glob.glob("%s\\%g_%g_*.dat" % (dest_dir, eta, eps))
    L = []
    for fin in files:
        L.append(int(os.path.basename(fin).rstrip(".dat").split("_")[2]))
    L = np.array(sorted(L))
    rate = np.zeros(L.size)
    for i in range(L.size):
        fin = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L[i])
        with open(fin, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if int(line.split("\t")[0]) == 2:
                    count += 1
            rate[i] = count / len(lines)
    return L, rate


if __name__ == "__main__":
    # for L in [2048]:
    #     eta = 0.18
    #     eps = 0.08
    #     # cal_phi_chi(L, eps, eta)
    #     phi_mean, phi_var, theta_mean, theta_var,\
    #         phi_mean2, phi_var2, theta_mean2, theta_var2 \
    #         = read_time_ave_phi(L, eps, eta)
    #     # plt.plot(theta_var, phi_mean, "o", label="%f" % (np.mean(phi_mean)))
    #     # plt.plot(
    #     #     theta_var2, phi_mean2, "s", label="%f" % (np.mean(phi_mean2)))
    #     # # plt.yscale("log")
    #     # plt.xscale("log")
    #     # plt.legend()
    #     # plt.show()
    #     # plt.close()

    #     # print(np.mean(phi_mean), np.mean(phi_mean2))
    #     # mask = phi_mean >= 0.67
    #     # print(np.mean(phi_mean), np.mean(phi_mean[mask]))

    #     bins = np.linspace(0.7, 0.8, 32)
    #     plt.hist(phi_mean2, bins, density=True, alpha=0.5, label=r"$L=%d$" % L)
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    disorder_t = "RT"
    L = 724
    eps = 0.03
    # cal_phi_chi(L, eps, 0.18, disorder_t)

    # if disorder_t == "RF":
    #     dest_dir = "D:\\data\\VM2d\\random_field\\phi_chi"
    # else:
    #     dest_dir = "D:\\data\\VM2d\\random_torque\\phi_chi"
    # fin = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L)
    # phi_mean, phi_var, theta_mean, theta_var,\
    #     phi_mean2, phi_var2, theta_mean2, theta_var2 = read_data(fin)
    # print(np.mean(phi_mean), np.mean(phi_mean2))
    # print(np.mean(phi_var), np.mean(phi_var2))
    # print(np.var(phi_mean), np.var(phi_mean2))

    L, r = get_nonsteady_rates(eps)
    for i in range(L.size):
        print(L[i], r[i])
    # plt.loglog(L, r, "o")
    # plt.show()
    # plt.close()
