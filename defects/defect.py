import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


def cal_defects(theta, lbox=4):
    def cal_dtheta(j1, i1, j2, i2):
        dtheta = theta[j1, i1] - theta[j2, i2]
        if dtheta > M_PI:
            dtheta -= M_2PI
        elif dtheta < -M_PI:
            dtheta += M_2PI
        return dtheta

    M_PI = np.pi
    M_2PI = np.pi * 2
    w1 = M_2PI - M_PI / 180
    w2 = M_2PI + M_PI / 180
    dict_charge = {1: {"x": [], "y": []}, -1: {"x": [], "y": []}}
    nrwos, ncols = theta.shape
    for j in range(nrwos):
        for i in range(ncols):
            w = cal_dtheta(j, i, j - 1, i) +\
                cal_dtheta(j, i - 1, j, i) +\
                cal_dtheta(j - 1, i - 1, j, i - 1) +\
                cal_dtheta(j - 1, i, j - 1, i - 1)
            if w1 < w < w2:
                dict_charge[1]["x"].append(i)
                dict_charge[1]["y"].append(j)
            elif -w2 < w < -w1:
                dict_charge[-1]["x"].append(i)
                dict_charge[-1]["y"].append(j)
    dict_charge[-1]["x"] = np.array([i for i in dict_charge[-1]["x"]]) * lbox
    dict_charge[-1]["y"] = np.array([i for i in dict_charge[-1]["y"]]) * lbox
    dict_charge[1]["x"] = np.array([i for i in dict_charge[1]["x"]]) * lbox
    dict_charge[1]["y"] = np.array([i for i in dict_charge[1]["y"]]) * lbox
    return dict_charge


def detect_defects(vx, vy, lbox=4):
    """ Find a bug that often 4 defects are found where there should be one
        defect instead.
    """
    def dtheta(row1, col1, row2, col2, theta):
        phi = theta[row1, col1] - theta[row2, col2]
        if phi > 180:
            phi -= 360
        elif phi < -180:
            phi += 360
        return phi

    nrows, ncols = vx.shape
    theta = np.arctan2(vy, vx) / np.pi * 180
    theta[theta < 0] += 360
    charge_dict = {1: {"x": [], "y": []}, -1: {"x": [], "y": []}}
    for row_c in range(nrows):
        row_1 = row_c + 1
        if row_1 >= nrows:
            row_1 = 0
        row_0 = row_c - 1
        if row_0 < 0:
            row_0 = nrows - 1
        for col_c in range(ncols):
            col_1 = col_c + 1
            if col_1 >= ncols:
                col_1 = 0
            col_0 = col_c - 1
            if col_0 < 0:
                col_0 = ncols - 1
            w = dtheta(row_1, col_c, row_1, col_1, theta) +\
                dtheta(row_1, col_0, row_1, col_c, theta) +\
                dtheta(row_c, col_0, row_1, col_0, theta) +\
                dtheta(row_0, col_0, row_c, col_0, theta) +\
                dtheta(row_0, col_c, row_0, col_0, theta) +\
                dtheta(row_0, col_1, row_0, col_c, theta) +\
                dtheta(row_c, col_1, row_0, col_1, theta) +\
                dtheta(row_1, col_1, row_c, col_1, theta)
            if 359 < w < 361:
                charge_dict[1]["x"].append(col_c)
                charge_dict[1]["y"].append(row_c)
            elif -361 < w < -359:
                charge_dict[-1]["x"].append(col_c)
                charge_dict[-1]["y"].append(row_c)

    charge_dict[1]["x"] = np.array([(i + 0.5) * lbox
                                    for i in charge_dict[1]["x"]])
    charge_dict[1]["y"] = np.array([(i + 0.5) * lbox
                                    for i in charge_dict[1]["y"]])
    charge_dict[-1]["x"] = np.array([(i + 0.5) * lbox
                                     for i in charge_dict[-1]["x"]])
    charge_dict[-1]["y"] = np.array([(i + 0.5) * lbox
                                     for i in charge_dict[-1]["y"]])
    return charge_dict


def plot_defects(ax, defects, ms=5, mew=2):
    ax.plot(defects[1]["x"],
            defects[1]["y"],
            "ko",
            fillstyle="none",
            ms=ms,
            mew=mew)
    ax.plot(defects[-1]["x"],
            defects[-1]["y"],
            "wo",
            fillstyle="none",
            ms=ms,
            mew=mew)


def compare_raw_smoothed(eps, L=512, eta=0.18):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,
                                        figsize=(15, 5.5),
                                        constrained_layout=True,
                                        sharey=True)
    ms = 5
    mew = 2
    fin = "time_ave/RT_ave_%d_%.3f_%.3f_30370000_200.npz" % (L, eta, eps)
    npfile = np.load(fin)
    vx = npfile["vx_m"]
    vy = npfile["vy_m"]
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    ax1.imshow(theta, cmap="hsv", origin="lower", extent=(0, L, 0, L))
    charges = cal_defects(theta)
    n_p = charges[1]["x"].size
    n_m = charges[-1]["x"].size
    ax1.plot(charges[1]["x"],
             charges[1]["y"],
             "o",
             c="k",
             ms=ms,
             fillstyle="none",
             mew=mew)
    ax1.plot(charges[-1]["x"],
             charges[-1]["y"],
             "o",
             c="w",
             ms=ms,
             fillstyle="none",
             mew=mew)
    ax1.set_title(r"Raw: $n_+=%d, n_-=%d$" % (n_p, n_m), fontsize="large")

    sigma = 0.8
    vx_s = gaussian_filter(vx, sigma=sigma, mode="wrap")
    vy_s = gaussian_filter(vy, sigma=sigma, mode="wrap")
    theta_s = np.arctan2(vy_s, vx_s)
    theta_s[theta_s < 0] += np.pi * 2
    charges_s = cal_defects(theta_s)
    ax2.imshow(theta_s, cmap="hsv", origin="lower", extent=(0, L, 0, L))
    ax2.plot(charges_s[1]["x"],
             charges_s[1]["y"],
             "o",
             c="k",
             ms=ms,
             fillstyle="none",
             mew=mew)
    ax2.plot(charges_s[-1]["x"],
             charges_s[-1]["y"],
             "o",
             c="w",
             ms=ms,
             fillstyle="none",
             mew=mew)
    n_p2 = charges_s[1]["x"].size
    n_m2 = charges_s[-1]["x"].size
    ax2.set_title(r"Gaussian smoothed $\sigma=0.8:\quad n_+=%d, n_-=%d$" %
                  (n_p2, n_m2))

    sigma = 1
    vx_s = gaussian_filter(vx, sigma=sigma, mode="wrap")
    vy_s = gaussian_filter(vy, sigma=sigma, mode="wrap")
    theta[theta < 0] += np.pi * 2
    theta_s = np.arctan2(vy_s, vx_s)
    theta_s[theta_s < 0] += np.pi * 2
    charges_s = cal_defects(theta_s)
    ax3.imshow(theta_s, cmap="hsv", origin="lower", extent=(0, L, 0, L))
    ax3.plot(charges_s[1]["x"],
             charges_s[1]["y"],
             "o",
             c="k",
             ms=ms,
             fillstyle="none",
             mew=mew)
    ax3.plot(charges_s[-1]["x"],
             charges_s[-1]["y"],
             "o",
             c="w",
             ms=ms,
             fillstyle="none",
             mew=mew)
    n_p3 = charges_s[1]["x"].size
    n_m3 = charges_s[-1]["x"].size
    ax3.set_title(r"Gaussian smoothed $\sigma=1.0\quad n_+=%d, n_-=%d$" %
                  (n_p3, n_m3))

    plt.suptitle(
        r"RT, long time-averaged momentum fields: $L=%d, \eta=%g, \epsilon=%g$"
        % (L, eta, eps),
        fontsize="x-large")
    plt.show()
    plt.close()


def get_long_time_ave_defect(eps,
                             L,
                             eta,
                             theta0,
                             seed,
                             smoothed=True,
                             lbox=4,
                             sigma=0.8,
                             ret_module=False):
    fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed,
                                                        theta0)
    npfile = np.load(fin)
    vx = npfile["vx_m"]
    vy = npfile["vy_m"]
    if smoothed:
        vx = gaussian_filter(vx, sigma, mode="wrap")
        vy = gaussian_filter(vy, sigma, mode="wrap")
    theta = np.arctan2(vy, vx)
    defects = cal_defects(theta, lbox)
    if not ret_module:
        return theta, defects
    else:
        module = np.sqrt(vx**2 + vy**2)
        return module, theta, defects


def get_defect_num(vx, vy, smoothed, lbox, sigma=0.8):
    if smoothed:
        vx = gaussian_filter(vx, sigma, mode="wrap")
        vy = gaussian_filter(vy, sigma, mode="wrap")
    theta = np.arctan2(vy, vx)
    defects = cal_defects(theta, lbox)
    return defects[1]["x"].size


def compare_short_long(eps, L=512, eta=0.18, smoothed=True, lbox=4):
    sigma = 1
    theta_lt, defects_lt = get_long_time_ave_defect(eps,
                                                    L,
                                                    eta,
                                                    100,
                                                    30370000,
                                                    True,
                                                    4,
                                                    sigma=sigma)
    theta_lt[theta_lt < 0] += np.pi * 2
    nd_lt = [defects_lt[1]["x"].size, defects_lt[-1]["x"].size]

    ms = 5
    fin = "time_ave_200/RT_ave_%d_%g_%g_200_30370000_100.bin" % (L, eta, eps)
    from decode import read_time_ave_bin
    frames = read_time_ave_bin(fin, beg=500)
    for frame in frames:
        rho, vx, vy = frame
        if smoothed:
            vx = gaussian_filter(vx, sigma, mode="wrap")
            vy = gaussian_filter(vy, sigma, mode="wrap")
        theta_st = np.arctan2(vy, vx)
        defects_st = cal_defects(theta_st, lbox)
        nd_st = [defects_st[1]["x"].size, defects_st[-1]["x"].size]
        fig, (ax1, ax2) = plt.subplots(ncols=2,
                                       figsize=(10, 5),
                                       constrained_layout=True)
        ax1.imshow(theta_lt, cmap="hsv", origin="lower", extent=(0, L, 0, L))
        ax1.plot(defects_lt[1]["x"], defects_lt[1]["y"], "o", c="k", ms=ms)
        ax1.plot(defects_lt[-1]["x"], defects_lt[-1]["y"], "o", c="w", ms=ms)
        ax2.imshow(theta_st, cmap="hsv", origin="lower", extent=(0, L, 0, L))
        ax2.plot(defects_st[1]["x"], defects_st[1]["y"], "o", c="k", ms=ms)
        ax2.plot(defects_st[-1]["x"], defects_st[-1]["y"], "o", c="w", ms=ms)
        ax1.set_title(r"Long-time averaged: $n_+=%d, n_-=%d$" %
                      (nd_lt[0], nd_lt[1]))
        ax2.set_title(r"Short-time averaged: $n_+=%d, n_-=%d$" %
                      (nd_st[0], nd_st[1]))

        plt.show()
        plt.close()


def cal_defect_num_serials(eps,
                           L=512,
                           eta=0.18,
                           lbox=4,
                           t_win=200,
                           seed=30370000,
                           theta0=100):
    fin = "time_ave_200/RT_ave_%d_%g_%g_200_%d_%d.bin" % (L, eta, eps, seed,
                                                          theta0)
    from decode import read_time_ave_bin
    frames = read_time_ave_bin(fin, beg=500)
    t_arr = []
    nd_arr = []
    nd_smoothed_arr1 = []
    nd_smoothed_arr2 = []
    t_win_min = 200
    rho, vx, vy = np.zeros((3, L // lbox, L // lbox))
    n_accum = t_win // t_win_min
    count_accum = 0
    for i, frame in enumerate(frames):
        rho_tmp, vx_tmp, vy_tmp = frame
        if t_win_min == t_win:
            rho, vx, vy = rho_tmp, vx_tmp, vy_tmp
        else:
            rho += rho_tmp
            vx += vx_tmp
            vy += vy_tmp
        count_accum += 1
        if count_accum == n_accum:
            count_accum = 0
            if t_win_min != t_win:
                rho /= n_accum
                vx /= n_accum
                vy /= n_accum
            t_arr.append(300000 + 500 * 200 + i * 200)
            theta = np.arctan2(vy, vx)
            defects = cal_defects(theta, lbox)
            nd_arr.append(defects[1]["x"].size)
            sigma = 0.8
            vx_s = gaussian_filter(vx, sigma, mode="wrap")
            vy_s = gaussian_filter(vy, sigma, mode="wrap")
            theta = np.arctan2(vy_s, vx_s)
            defects = cal_defects(theta, lbox)
            nd_smoothed_arr1.append(defects[1]["x"].size)
            sigma = 1.0
            vx_s = gaussian_filter(vx, sigma, mode="wrap")
            vy_s = gaussian_filter(vy, sigma, mode="wrap")
            theta = np.arctan2(vy_s, vx_s)
            defects = cal_defects(theta, lbox)
            nd_smoothed_arr2.append(defects[1]["x"].size)
    fout = "defects_serials/RT_ave_%d_%.3f_%.3f_%d_%d_%d.dat" % (
        L, eta, eps, t_win, seed, theta0)
    with open(fout, "w") as f:
        for i in range(len(t_arr)):
            f.write("%d\t%d\t%d\t%d\n" %
                    (t_arr[i], nd_arr[i], nd_smoothed_arr1[i],
                     nd_smoothed_arr2[i]))


def cal_all_defects_serials():
    L = 512
    eta = 0.18
    lbox = 4
    t_win = 200
    seed = 30370000
    theta0 = 100
    eps_arr = np.array([
        0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
        0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25,
        0.26, 0.27, 0.28, 0.29, 0.30
    ])

    t_win_arr = [200, 1000, 5000, 25000]
    """
    eps_arr = np.array([0.05, 0.06])

    t_win_arr = [
        200, 400, 600, 800, 1000, 1400, 2000, 3200, 5000, 10000, 16000, 25000,
        40000
    ]
    t_win_arr = [60000, 80000, 100000, 140000, 200000]
    """

    for eps in eps_arr:
        for t_win in t_win_arr:
            print("t_win=%d, eps=%.3f" % (t_win, eps))
            cal_defect_num_serials(eps, L, eta, lbox, t_win, seed, theta0)


def read_defect_num(L, eta, eps, seed, theta0, t_win, lbox=4):
    if t_win is None:
        fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%d.npz" % (L, eta, eps, seed,
                                                          theta0)
        npfile = np.load(fin)
        vx = npfile["vx_m"]
        vy = npfile["vy_m"]
        theta = np.arctan2(vy, vx)
        sigma = 0.8
        vx_s = gaussian_filter(vx, sigma=sigma, mode="wrap")
        vy_s = gaussian_filter(vy, sigma=sigma, mode="wrap")
        theta_s = np.arctan2(vy_s, vx_s)
        charges = cal_defects(theta)
        charges_s = cal_defects(theta_s)
        nd_raw = charges[1]["x"].size
        nd_s = charges_s[1]["x"].size
        return nd_raw, nd_s
    else:
        fin = "defects_serials/RT_ave_%d_%.3f_%.3f_%d_%d_%d.dat" % (
            L, eta, eps, t_win, seed, theta0)
        with open(fin, "r") as f:
            lines = f.readlines()
            n1, n2, n3 = 0, 0, 0
            for line in lines:
                s = line.rstrip("\n").split("\t")
                n1 += int(s[1])
                n2 += int(s[2])
                n3 += int(s[3])
            nd_raw = n1 / len(lines)
            nd_s1 = n2 / len(lines)
            nd_s2 = n3 / len(lines)
        return nd_raw, nd_s1, nd_s2


def plot_defect_num_vs_eps(L=512, eta=0.18, seed=30370000, theta0=100, lbox=4):
    """ Plot defect number vs. epsilon """
    eps_arr = np.array([
        0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
        0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25,
        0.26, 0.27, 0.28, 0.29, 0.30
    ])

    t_win_arr = [200, 1000, 5000, 25000]
    ms = 5
    for t_win in t_win_arr:
        n1_arr, n2_arr, n3_arr = np.zeros((3, eps_arr.size))
        for i, eps in enumerate(eps_arr):
            n1_arr[i], n2_arr[i], n3_arr[3] = read_defect_num(
                L, eta, eps, seed, theta0, t_win, lbox)
        line, = plt.plot(eps_arr, n1_arr, "o", fillstyle="none", ms=ms)
        if t_win is None:
            label = r"$10^6$"
        else:
            label = r"$%d$" % t_win
        plt.plot(eps_arr, n2_arr, "o", c=line.get_c(), ms=ms, label=label)
    plt.legend(title=r"$\Delta t=$")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("defect number")
    plt.title(r"RT: $\eta=%g, L=%d$" % (eta, L))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_defect_num_vs_t_win(L=512,
                             eta=0.18,
                             seed=30370000,
                             theta0=100,
                             lbox=4):
    """ Plot defect number vs. t_win """
    """ Plot defect number vs. epsilon """
    # eps_arr = np.array([0.055, 0.15, 0.28])
    eps_arr = np.array([0.05, 0.06, 0.15])
    '''
    t_win_arr = [
        200, 400, 600, 800, 1000, 1400, 2000, 3200, 5000, 10000, 16000, 25000,
        40000, 60000, 80000, 100000, 140000, 200000
    ]
    '''
    t_win_arr = [
        200, 400, 600, 800, 1000, 1400, 2000, 3200, 5000, 10000, 16000, 25000
    ]
    ms = 5
    for eps in eps_arr:
        n1_arr, n2_arr, n3_arr = np.zeros((3, len(t_win_arr)))
        for i, t_win in enumerate(t_win_arr):
            n1_arr[i], n2_arr[i], n3_arr[i] = read_defect_num(
                L, eta, eps, seed, theta0, t_win, lbox)
        line, = plt.plot(t_win_arr, n1_arr, "o", fillstyle="none", ms=ms)
        label = "%.3f" % eps
        plt.plot(t_win_arr,
                 n2_arr,
                 "s",
                 fillstyle="none",
                 c=line.get_c(),
                 ms=ms,
                 label=label)
        plt.plot(t_win_arr,
                 n3_arr,
                 "^",
                 fillstyle="none",
                 c=line.get_c(),
                 ms=ms)
    plt.legend(title=r"$\epsilon=$")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("defect number")
    plt.xscale("log")
    # plt.yscale("log")
    plt.title(r"RT: $\eta=%g, L=%d$" % (eta, L))
    plt.tight_layout()
    plt.show()
    plt.close()


def dynamic_pinned_defects(eps, L=512, eta=0.18, smoothed=True, lbox=4):
    sigma = 1
    theta_lt, defects_lt = get_long_time_ave_defect(eps, L, eta, 100, 30370000,
                                                    True, 4, sigma)
    ms = 5
    fin = "time_ave_200/RT_ave_%d_%g_%g_200_30370000_100.bin" % (L, eta, eps)
    from decode import read_time_ave_bin
    frames = read_time_ave_bin(fin, beg=500)
    vx, vy = np.zeros((2, L // lbox, L // lbox))
    period = 100
    for i, frame in enumerate(frames):
        rho, vx_i, vy_i = frame
        vx += vx_i
        vy += vy_i
        if (i + 1) % period == 0:
            vx /= period
            vy /= period
            if smoothed:
                vx = gaussian_filter(vx, sigma, mode="wrap")
                vy = gaussian_filter(vy, sigma, mode="wrap")
            theta_st = np.arctan2(vy, vx)
            defects_st = cal_defects(theta_st, lbox)
            plt.plot(defects_lt[1]["x"],
                     defects_lt[1]["y"],
                     "o",
                     c="tab:red",
                     ms=ms,
                     alpha=0.5)
            plt.plot(defects_lt[-1]["x"],
                     defects_lt[-1]["y"],
                     "o",
                     c="tab:blue",
                     ms=ms,
                     alpha=0.5)
            plt.plot(defects_st[1]["x"],
                     defects_st[1]["y"],
                     "ro",
                     ms=ms,
                     fillstyle="none")
            plt.plot(defects_st[-1]["x"],
                     defects_st[-1]["y"],
                     "bo",
                     ms=ms,
                     fillstyle="none")
            plt.axis("scaled")
            plt.xlim(0, L // 4)
            plt.ylim(0, L // 4)
            plt.show()
            plt.close()
            vx, vy = np.zeros((2, L // lbox, L // lbox))


if __name__ == "__main__":
    os.chdir("E:/data/random_torque/defects/L=512_seed=30370000")
    L = 512
    eta = 0.18
    # eps = 0.15
    # fin = "time_ave/RT_ave_%d_%.3f_%.3f_30370000_100.npz" % (L, eta, eps)
    # npfile = np.load(fin)
    # vx = npfile["vx_m"]
    # vy = npfile["vy_m"]

    # sigma = 0.8
    # vx = gaussian_filter(vx, sigma=sigma, mode="wrap")
    # vy = gaussian_filter(vy, sigma=sigma, mode="wrap")
    # theta = np.arctan2(vy, vx)
    # theta[theta < 0] += np.pi * 2
    # plt.imshow(theta, cmap="hsv", origin="lower", extent=(0, L, 0, L))

    # charges = cal_defect(vx, vy)
    # np = charges[1]["x"].size
    # nm = charges[-1]["x"].size
    # plt.plot(charges[1]["x"], charges[1]["y"], "o", c="k", ms=5)
    # plt.plot(charges[-1]["x"], charges[-1]["y"], "s", c="w", ms=5)
    # plt.title(r"$n_+=%d, n_-=%d$" % (np, nm))
    # plt.show()
    # plt.close()

    # compare_raw_smoothed(eps=0.06)

    # plot_defect_num_vs_eps()
    # plot_defect_num_vs_t_win()
    # compare_short_long(0.06)
    # get_defect_num_serials(0.055, t_win=10000)
    cal_all_defects_serials()
    # dynamic_pinned_defects(0.15)
