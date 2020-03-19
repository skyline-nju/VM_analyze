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
    theta_var = np.var(theta[ncut0:])

    x = np.arange(phi_arr.size)
    axes[0].plot(x, phi_arr, label="mean=%f, var=%f" % (phi_mean, phi_var))
    axes[1].plot(x, theta, label="var=%f" % theta_var)
    axes[0].axvline(ncut0, c="tab:red")
    axes[1].axvline(ncut0, c="tab:red")
    axes[0].set_ylabel(r"$m$")
    axes[1].set_ylabel(r"$\theta$")
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
            theta_var = np.var(theta[beg:end])
            axes[0].axvspan(
                beg,
                end,
                alpha=0.4,
                label="mean=%f, var=%f" % (phi_mean, phi_var))
            axes[1].axvspan(beg, end, alpha=0.4, label="var=%f" % theta_var)
        axes[0].legend()
        axes[1].legend()
        plt.savefig("%s\\%g_%g_%d_%s.png" % (out_dir, eta, eps, L, seed))
    plt.close()
    return phi_mean, phi_var, theta_var


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
        pass

    seed_list = []
    dest_file = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L)
    if os.path.exists(dest_file):
        with open(dest_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                seed_list.append(int(line.split("\t")[0]))
        fout = open(dest_file, "a")
    else:
        fout = open(dest_file, "w")
    if L > 1000:
        ncut0 = 4000
    else:
        ncut0 = 3000
    fig_dir = "%s\\fig" % (dest_dir)
    for i, fin in enumerate(files):
        seed = fin.rstrip(".dat").split(para_sep)[-1]
        if int(seed) not in seed_list:
            savefig = False
            phi, theta = read_phi_theta(fin, 0)
            phi_mean, phi_var, theta_var = plot_serials(
                L, eps, eta, seed, disorder_t, phi, theta, ncut0)
            str_in = input("%d/%d: please set beg, end: " % (i, len(files)))
            if str_in == "":
                beg = ncut0
                end = phi.size
            else:
                str_list = str_in.split(" ")
                if len(str_list) == 1:
                    if str_list[0] == "n":
                        beg = None
                        end = None
                        savefig = True
                    elif str_list[0] == "q":
                        break
                    else:
                        beg = int(str_list[0]) * 1000
                        end = phi.size
                        savefig = True
                else:
                    beg = int(str_list[0]) * 1000
                    end = int(str_list[1]) * 1000
                    savefig = True
            if savefig:
                phi_mean2, phi_var2, theta_var2 = plot_serials(
                    L, eps, eta, seed, disorder_t, phi, theta, ncut0, fig_dir,
                    beg, end)
                if beg is None:
                    line = "%s\t%.8f\t%.8f\t%.8f\n" % (seed, phi_mean, phi_var,
                                                       theta_var)
                else:
                    line = "%s\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
                        seed, phi_mean, phi_var, theta_var, phi_mean2,
                        phi_var2, theta_var2)
            else:
                line = "%s\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
                    seed, phi_mean, phi_var, theta_var, phi_mean, phi_var,
                    theta_var)
            fout.write(line)
            if i % 10 == 0:
                fout.flush()
                os.fsync(fout)
    fout.close()


def read_time_ave_phi(L, eps, eta=0.18, disorder_t="RF"):
    if disorder_t == "RF":
        dest_dir = "D:\\data\\VM2d\\random_field\\phi_chi"
    else:
        pass
    fin = "%s\\%g_%g_%d.dat" % (dest_dir, eta, eps, L)
    with open(fin, "r") as f:
        lines = f.readlines()
        phi_mean, phi_var, theta_var, phi_mean2, phi_var2, theta_var2 = np.zeros(
            (6, len(lines)))
        col_len = np.zeros(len(lines), int)
        for i, line in enumerate(lines):
            s = line.rstrip("\n").split("\t")
            if len(s) == 4:
                col_len[i] = 4
                phi_mean[i], phi_var[i], theta_var[i] = float(s[1]), float(
                    s[2]), float(s[3])
            elif len(s) == 7:
                col_len[i] = 7
                phi_mean[i], phi_var[i], theta_var[i] = float(s[1]), float(
                    s[2]), float(s[3])
                phi_mean2[i], phi_var2[i], theta_var2[i] = float(s[4]), float(
                    s[5]), float(s[6])
        mask = col_len == 7
        phi_mean2, phi_var2, theta_var2 = phi_mean2[mask], phi_var2[
            mask], theta_var2[mask]
        return phi_mean, phi_var, theta_var, phi_mean2, phi_var2, theta_var2


def get_sample_ave_phi(L, eps, eta=0.18, disorder_t="RF"):
    phi_mean, phi_var, theta_var, phi_mean2, phi_var2, theta_var2 \
        = read_time_ave_phi(L, eps, eta)
    phi = np.mean(phi_mean)
    phi2 = np.mean(phi_mean2)
    chi_dis = np.var(phi_mean) * L ** 2
    chi_dis2 = np.var(phi_mean2) * L ** 2
    chi_con = np.mean(phi_var) * L ** 2
    chi_con2 = np.mean(phi_var2) * L ** 2
    return phi, chi_dis, chi_con, phi2, chi_dis2, chi_con2


if __name__ == "__main__":
    for L in [724, 1024, 1448, 2048]:
        eta = 0.18
        eps = 0.09
        # cal_phi_chi(L, eps, eta)
        phi_mean, phi_var, theta_var, phi_mean2, phi_var2, theta_var2 \
            = read_time_ave_phi(L, eps, eta)
        # plt.plot(theta_var, phi_mean, "o", label="%f" % (np.mean(phi_mean)))
        # plt.plot(
        #     theta_var2, phi_mean2, "s", label="%f" % (np.mean(phi_mean2)))
        # # plt.yscale("log")
        # plt.xscale("log")
        # plt.legend()
        # plt.show()
        # plt.close()

        # print(np.mean(phi_mean), np.mean(phi_mean2))
        # mask = phi_mean >= 0.67
        # print(np.mean(phi_mean), np.mean(phi_mean[mask]))

        bins = np.linspace(0.63, 0.78, 32)
        plt.hist(phi_mean2, bins, density=True, alpha=0.5, label=r"$L=%d$" % L)
    plt.yscale("log")
    plt.legend()
    plt.show()