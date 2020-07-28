import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from read_bin import get_time_averaged_image, read_bin


def get_theta(vx, vy):
    theta = np.arctan2(vy, vx) / np.pi * 180
    theta[theta < 0] += 360
    return theta


def cal_defect(theta, bins=4):
    def angle(j1, i1, j2, i2):
        dtheta = int(theta[j1, i1]) - int(theta[j2, i2])
        if dtheta > 180:
            dtheta -= 360
        elif dtheta < -180:
            dtheta += 360
        return dtheta

    nrows, ncols = theta.shape
    S = np.zeros((nrows, ncols), int)
    for j in range(nrows):
        for i in range(ncols):
            S[j, i] = angle(j, i, j - 1, i) + angle(j, i - 1, j, i) + angle(
                j - 1, i - 1, j, i - 1) + angle(j - 1, i, j - 1, i - 1)
    dict_charge = {1: {"row": [], "col": []}, -1: {"row": [], "col": []}}
    for j in range(nrows):
        for i in range(ncols):
            if S[j, i] == 360:
                dict_charge[1]["row"].append(j)
                dict_charge[1]["col"].append(i)
            elif S[j, i] == -360:
                dict_charge[-1]["row"].append(j)
                dict_charge[-1]["col"].append(i)
    dict_charge[-1]["row"] = np.array([i
                                       for i in dict_charge[-1]["row"]]) * bins
    dict_charge[-1]["col"] = np.array([i
                                       for i in dict_charge[-1]["col"]]) * bins
    dict_charge[1]["row"] = np.array([i for i in dict_charge[1]["row"]]) * bins
    dict_charge[1]["col"] = np.array([i for i in dict_charge[1]["col"]]) * bins
    return dict_charge


def get_snap_file(L, eps, eta, seed, seed2, disorder_t, ic="ordered"):
    if seed == 30370020:
        t_win = 2000
    else:
        t_win = 200
    if L == 1024:
        t_win = 10000
    elif L == 2048 or L == 4096:
        t_win = 50000
    if disorder_t == "RT":
        folder = r"E:\data\random_torque\replica"
        if ic == "ordered_sym":
            bin_dir = r"%s\ini_ordered_sym" % folder
            t_win = 10000
            fname = "%s\\RT_ave_%d_%g_%g_%d_%d_%03d.bin" % (
                bin_dir, L, eta, eps, t_win, seed, seed2)
            if L == 2048:
                t_win = 50000
                fname = "%s\\RT_ave_%d_%g_%g_%d_%d_%d.bin" % (
                    bin_dir, L, eta, eps, t_win, seed, seed2)
        elif ic == "ordered":
            new_seeds = [30370052]
            if seed in new_seeds:
                bin_dir = r"%s\ini_ordered_new" % folder
                t_win = 10000
                fname = "%s\\RT_ave_%d_%g_%g_%d_%d_%03d.bin" % (
                    bin_dir, L, eta, eps, t_win, seed, seed2)
        elif ic == "ordered_diag":
            bin_dir = r"%s\ini_ordered_diag" % folder
            t_win = 10000
            fname = "%s\\RT_ave_%d_%g_%g_%d_%d_%03d.bin" % (
                bin_dir, L, eta, eps, t_win, seed, seed2)
        else:
            bin_dir = "%s\\time_ave" % folder
            fname = "%s\\RT_ave_%d_%g_%g_%d_%d_%d.bin" % (bin_dir, L, eta, eps,
                                                          t_win, seed, seed2)
    else:
        bin_dir = r"D:\data\VM2d\random_field\time_ave_image"
        fname = "%s\\RF_ave_%d_%g_%g_%d_%d_%d.bin" % (bin_dir, L, eta, eps,
                                                      t_win, seed, seed2)
    return fname


def plot_defect(L,
                eps,
                eta=0.18,
                seed=30370000,
                seed2=100,
                disorder_t="RT",
                ic="ordered",
                ax=None):
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    else:
        flag_show = False
    beg, end = 100, None
    fname = get_snap_file(L, eps, eta, seed, seed2, disorder_t, ic)
    vx, vy = get_time_averaged_image(fname, beg, end, "v", 128)
    # sigma = 0.8
    # vx2 = gaussian_filter(vx, sigma=sigma, mode="wrap")
    # vy2 = gaussian_filter(vy, sigma=sigma, mode="wrap")
    # theta = get_theta(vx2, vy2)
    theta = get_theta(vx, vy)
    charge = cal_defect(theta)
    im = ax.imshow(
        theta,
        cmap="hsv",
        origin="lower",
        extent=[0, L, 0, L],
        vmin=0,
        vmax=360)
    ax.plot(charge[1]["col"], charge[1]["row"], "wo", fillstyle="none", mew=2)
    ax.plot(
        charge[-1]["col"], charge[-1]["row"], "ks", fillstyle="none", mew=2)
    ax.set_xticks([0, 256, 512])
    ax.set_yticks([0, 256, 512])
    ax.set_title(
        r"$n^+=%d, n^-=%d$" % (charge[1]["col"].size, charge[-1]["col"].size))
    if flag_show:
        plt.show()
        plt.close()
    return im


def plot_defect_diff_ini_condi(L,
                               eps,
                               eta=0.18,
                               seed=30370000,
                               disorder_t="RT",
                               ic="ordered"):
    seed1 = 100
    seed2 = 200
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
    plot_defect(L, eps, eta, seed, seed1, disorder_t, ic, ax1)
    im = plot_defect(L, eps, eta, seed, seed2, disorder_t, ic, ax2)
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    plt.suptitle(title, fontsize="x-large")
    cb = plt.colorbar(im, ax=ax2)
    cb_ticks = [0, 90, 180, 270, 360]
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels([r"$%d\degree$" % i for i in cb_ticks])
    plt.show()
    plt.close()


def plot_rho(L, eps, eta=0.18, seed=30370020, disorder_t="RT", ic="ordered"):
    seed2_arr = [0, 270]
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), constrained_layout=True)
    beg, end = 0, None
    for i, seed2 in enumerate(seed2_arr):
        fname = get_snap_file(L, eps, eta, seed, seed2, disorder_t, ic)
        rho_mean = get_time_averaged_image(fname, beg, end, "rho", L // 4)
        im = ax[i].imshow(
            rho_mean, origin="lower", extent=[0, L, 0, L], vmin=0, vmax=2)
    ax[0].set_title("replica 0")
    ax[1].set_title("replica 3")
    plt.colorbar(im, ax=ax[1], extend="max")
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$, Sample B" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    plt.suptitle(title, fontsize="x-large")
    plt.show()
    plt.close()


def plot_momentum(L,
                  eps,
                  eta=0.18,
                  seed=2022324,
                  disorder_t="RT",
                  ic="ordered",
                  seed2=180,
                  idx=None):
    from matplotlib.colors import hsv_to_rgb

    def get_rgb(theta, module, m_max=None):
        H = theta / 360
        V = module
        if m_max is not None:
            V[V > m_max] = m_max
        S = np.ones_like(H)
        HSV = np.dstack((H, S, V))
        RGB = hsv_to_rgb(HSV)
        return RGB

    def add_colorbar(ax, mmin, mmax):
        V, H = np.mgrid[0:1:50j, 0:1:180j]
        S = np.ones_like(V)
        HSV = np.dstack((H, S, V))
        RGB = hsv_to_rgb(HSV)
        extent = [0, 360, mmin, mmax]
        ax.imshow(RGB, origin='lower', extent=extent, aspect='auto')
        ax.set_xlabel('orientation', fontsize="large")
        ticks = [0, 90, 180, 270, 360]
        ticklabels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_ylabel('module', fontsize="large")

    beg, end = 10, None
    if idx is not None:
        seed2_list = [0, 120, 180, 240, 300, 60]
        seed2 = seed2_list[idx]
    fname = get_snap_file(L, eps, eta, seed, seed2, disorder_t, ic)
    n_bins = L // 4

    rho, vx, vy = get_time_averaged_image(fname, beg, end, "both", n_bins)
    fig, ax = plt.subplots(figsize=(5, 6))

    box = [0, L, 0, L]
    v_orient = np.arctan2(vy, vx) / np.pi * 180
    v_orient[v_orient < 0] += 360
    v_module = np.sqrt(vx**2 + vy**2)
    module_max = 4
    RGB = get_rgb(v_orient, v_module, m_max=module_max)
    ax.imshow(RGB, extent=box, origin="lower")
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    # title += ", Sample A, replica %d" % idx

    ax.set_title(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    bbox = ax.get_position().get_points().flatten()
    cax = plt.axes([bbox[0], 0.075, bbox[2] - bbox[0], 0.1])
    add_colorbar(cax, 0, module_max)

    plt.show()
    plt.close()


def plot_momentum_hist(L, eps, eta=0.18, seed=20203262, disorder_t="RT"):
    if seed == 20203262:
        seed2_list = [120, 180, 300]
        idx_list = [1, 2, 4]
    elif seed == 2022324:
        seed2_list = [0, 120]
        idx_list = [0, 1]
    beg, end = 10, None
    for i, seed2 in enumerate(seed2_list):
        fname = get_snap_file(L, eps, eta, seed, seed2, disorder_t)
        n_bins = L // 4
        rho, vx, vy = get_time_averaged_image(fname, beg, end, "both", n_bins)
        v_orient = np.arctan2(vy, vx) / np.pi
        v_orient[v_orient < 0] += 2
        v_module = np.sqrt(vx**2 + vy**2)
        bins = 72
        hist, bin_edges = np.histogram(
            v_orient, bins, (0, 2), density=True, weights=v_module)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        line, = plt.plot(
            x, hist, "-o", label="replica %d" % idx_list[i], fillstyle="none")
        hist, bin_edges = np.histogram(v_orient, bins, (0, 2), density=True)
        plt.plot(x, hist, "s", fillstyle="none", c=line.get_c())
    plt.xlabel(r"$\theta/\pi$", fontsize="large")
    plt.ylabel("PDF", fontsize="large")
    plt.legend()
    if disorder_t == "RT":
        title = r"RS: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    else:
        title = r"RF: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    title += ", Sample B"
    plt.title(title, fontsize="x-large")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_defect_short_win(L,
                          eps,
                          eta=0.18,
                          seed=30370000,
                          seed2=100,
                          disorder_t="RT"):
    fname = get_snap_file(L, eps, eta, seed, seed2, disorder_t)
    beg, end = 0, None
    vx, vy = get_time_averaged_image(fname, beg, end, "v", 128)
    sigma = 0.8
    vx2 = gaussian_filter(vx, sigma=sigma, mode="wrap")
    vy2 = gaussian_filter(vy, sigma=sigma, mode="wrap")
    theta = get_theta(vx2, vy2)
    charge = cal_defect(theta)

    beg = 100
    count = 0
    n_sum = 0
    frames = read_bin(fname, beg, end)
    for i, frame in enumerate(frames):
        n_mean, vx_mean, vy_mean = frame
        vx2 = gaussian_filter(vx_mean, sigma=1, mode="wrap")
        vy2 = gaussian_filter(vy_mean, sigma=1, mode="wrap")
        theta = get_theta(vx2, vy2)
        charge_t = cal_defect(theta)
        t = 300000 + (beg + i) * 200
        n_p_t = charge_t[1]["col"].size
        n_p = charge[1]["col"].size
        plt.subplots(figsize=(8, 6))
        im = plt.imshow(
            theta,
            cmap="hsv",
            origin="lower",
            extent=[0, L, 0, L],
            vmin=0,
            vmax=360)
        plt.plot(
            charge[1]["col"], charge[1]["row"], "wo", fillstyle="none", mew=2)
        plt.plot(
            charge[-1]["col"],
            charge[-1]["row"],
            "ks",
            fillstyle="none",
            mew=2)
        plt.plot(
            charge_t[1]["col"],
            charge_t[1]["row"],
            "wx",
            fillstyle="none",
            mew=2)
        plt.plot(
            charge_t[-1]["col"],
            charge_t[-1]["row"],
            "kx",
            fillstyle="none",
            mew=2)
        plt.xticks([0, 256, 512])
        plt.yticks([0, 256, 512])
        plt.title(r"$\Delta t=200, t=%d, n^{+/-}_t=%d$" % (t, n_p_t))
        cb = plt.colorbar(im)
        cb_ticks = [0, 90, 180, 270, 360]
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels([r"$%d\degree$" % i for i in cb_ticks])
        plt.tight_layout(rect=[0, -0.03, 1, 0.98])
        plt.suptitle(
            r"RS: $L=%d, \eta=%g, \epsilon=%g, n^{+/-}=%d$" % (L, eta, eps,
                                                               n_p),
            y=0.995)
        plt.show()
        plt.close()
        n_sum += n_p_t
        count += 1
    print(n_sum / count, n_p)


if __name__ == "__main__":
    plot_defect_diff_ini_condi(512, 0.06, disorder_t="RT", ic="rand")
    # plot_defect_short_win(512, 0.035, 0.18, seed2=200)
    # plot_rho(
    #     512,
    #     0.045,
    #     eta=0.18,
    #     disorder_t="RT",
    #     seed=25650015,
    #     ic="ordered_diag")
    # plot_momentum(
    #     512, 0.055, eta=0.18, disorder_t="RT", seed=30370000, seed2=500, ic="rand")
    # plot_momentum_hist(4096, 0.035, disorder_t="RT", seed=20203261)
