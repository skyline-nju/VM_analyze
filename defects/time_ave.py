import matplotlib.pyplot as plt
import numpy as np
import os
import decode
import time_corr as tc
from scipy.ndimage import gaussian_filter
import glob


def time_ave(fin, frame_beg=0, frame_end=None, L=512, lbox=4):
    """ Get time-averaged density and momentum fields. """
    frames = decode.read_time_ave_bin(fin, frame_beg, frame_end)
    rho_m = np.zeros((L // lbox, L // lbox))
    vx_m = np.zeros_like(rho_m)
    vy_m = np.zeros_like(rho_m)
    count = 0
    for frame in frames:
        rho, vx, vy = frame
        rho_m += rho
        vx_m += vx
        vy_m += vy
        count += 1
    rho_m /= count
    vx_m /= count
    vy_m /= count
    print(count)
    return rho_m, vx_m, vy_m


def get_fout(fin):
    basename = os.path.basename(fin)
    s = basename.rstrip(".bin").split("_")
    eta = float(s[3])
    eps = float(s[4])
    fout = r"time_ave/RT_ave_%s_%.3f_%.3f_%s_%s.npz" % (s[2], eta, eps, s[6],
                                                        s[7])
    return fout


def get_all_time_ave(t_win0=200):
    # files_list1 = [
    #     "RT_ave_512_0.180_0.060_200_30370000_000.bin",
    #     "RT_ave_512_0.180_0.060_200_30370000_090.bin",
    #     "RT_ave_512_0.180_0.060_200_30370000_180.bin",
    #     "RT_ave_512_0.180_0.060_200_30370000_270.bin"
    # ]
    # files = ["time_ave_200/%s" % i for i in files_list1]
    files = glob.glob("time_ave_%d/*.bin" % t_win0)
    for fin in files:
        # basename = os.path.basename(fin)
        # eps = float(basename.split("_")[4])
        # if eps <= 0.29:
        #     continue
        rho_m, vx_m, vy_m = time_ave(fin, 0)
        fout = get_fout(fin)
        np.savez_compressed(fout, rho_m=rho_m, vx_m=vx_m, vy_m=vy_m)


def plot_time_ave_density(L, eta, eps, seed, theta0):
    fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%d.npz" % (L, eta, eps, seed,
                                                      theta0)
    data = np.load(fin)
    rho_m = data["rho_m"]
    plt.imshow(rho_m, origin="lower", extent=[0, L, 0, L])
    plt.colorbar()
    plt.show()
    plt.close()


def plot_time_ave_momentum(L, eta, eps, seed, theta0):
    fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed,
                                                        theta0)
    data = np.load(fin)
    vx_m = data["vx_m"]
    vy_m = data["vy_m"]
    sigma = 1
    vx_m = gaussian_filter(vx_m, sigma=sigma, mode="wrap")
    vy_m = gaussian_filter(vy_m, sigma=sigma, mode="wrap")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    module = np.sqrt(vx_m**2 + vy_m**2)
    theta = np.arctan2(vy_m, vx_m)
    theta[theta < 0] += np.pi * 2
    import defect
    defects = defect.cal_defects(theta, 4)
    defect.plot_defects(ax1, defects, ms=2, mew=1)
    defect.plot_defects(ax2, defects, ms=2, mew=1)

    box = [0, L, 0, L]
    im1 = ax1.imshow(module, origin="lower", extent=box)
    im2 = ax2.imshow(theta, origin="lower", extent=box, cmap="hsv")
    ax1.set_title("(a) momentum module")
    ax2.set_title("(b) momentum orientation")
    plt.colorbar(im1, ax=ax1, orientation="vertical")
    plt.colorbar(im2, ax=ax2, orientation="vertical")
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    title = r"$L=%d, \eta=%g, \epsilon=%g, n_+=%d, n_-=%d$" % (
        L, eta, eps, defects[1]["x"].size, defects[-1]["x"].size)
    fig.suptitle(title, y=0.995, fontsize="x-large")
    plt.show()
    plt.close()


def plot_time_ave_density_momentum(L, eta, eps, seed, theta0):
    fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed,
                                                        theta0)
    data = np.load(fin)
    rho_m, vx_m, vy_m = data["rho_m"], data["vx_m"], data["vy_m"]
    sigma = 1
    vx_m = gaussian_filter(vx_m, sigma=sigma, mode="wrap")
    vy_m = gaussian_filter(vy_m, sigma=sigma, mode="wrap")
    module = np.sqrt(vx_m**2 + vy_m**2)
    theta = np.arctan2(vy_m, vx_m)
    theta[theta < 0] += np.pi * 2
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5.5))
    box = [0, L, 0, L]
    im1 = ax1.imshow(rho_m, origin="lower", extent=box)
    im2 = ax2.imshow(module, origin="lower", extent=box)
    im3 = ax3.imshow(theta, origin="lower", extent=box, cmap="hsv")

    plt.colorbar(im1, ax=ax1, orientation="horizontal")
    plt.colorbar(im2, ax=ax2, orientation="horizontal")
    plt.colorbar(im3, ax=ax3, orientation="horizontal")

    import defect
    defects = defect.cal_defects(theta, 4)
    # defect.plot_defects(ax1, defects, ms=1, mew=0.2)
    defect.plot_defects(ax2, defects, ms=4, mew=1)
    defect.plot_defects(ax3, defects, ms=4, mew=1)

    ax1.set_title("(a) density")
    ax2.set_title("(b) momentum module")
    ax3.set_title("(c) momentum orientation")
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    title = r"$L=%d, \eta=%g, \epsilon=%g, n_+=%d, n_-=%d$" % (
        L, eta, eps, defects[1]["x"].size, defects[-1]["x"].size)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title, y=0.995, fontsize="x-large")
    plt.show()
    plt.close()


def get_momentum_serials(fname, beg=0, end=None, lbox=4, L=512):
    n_frame = decode.get_nframe(fname, lbox) - beg
    nx, ny = L // lbox, L // lbox
    vx = np.zeros((n_frame, nx, ny), np.float32)
    vy = np.zeros_like(vx)
    frames = decode.read_time_ave_bin(fname, beg, end, "v", lbox)
    for i, (vx_i, vy_i) in enumerate(frames):
        vx[i] = vx_i
        vy[i] = vy_i
    return vx, vy


def cal_all_EA_OP(L, eta, seed, theta0, t_win0):
    eps_arr = np.array([
        0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055,
        0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
        0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
        0.30
    ])

    # eps_arr = np.array([0.001, 0.002, 0.003, 0.005, 0.01, 0.04])
    for eps in eps_arr:
        fin = "time_ave_%d/RT_ave_%d_%g_%g_%d_%d_%d.bin" % (
            t_win0, L, eta, eps, t_win0, seed, theta0)
        print(fin)
        fout = "EA_OP/%d_%.3f_%.3f_%d_%d_%d.npz" % (L, eta, eps, t_win0, seed,
                                                    theta0)
        vx, vy = get_momentum_serials(fin, 500, lbox=4, L=L)
        EA_OP_t, EA_OP_r = tc.cal_EA_OP(vx, vy)
        np.savez_compressed(fout, time_serials=EA_OP_t, last_frame=EA_OP_r)


def plot_EA_OP_vs_eps(L, eta, seed, theta0, t_win0, ax=None, show_phi=False):
    eps_arr = np.array([
        0.001, 0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06,
        0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30
    ])
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    else:
        flag_show = False
    EA_OP_arr = np.zeros_like(eps_arr)
    for i, eps in enumerate(eps_arr):
        fin = "EA_OP/%d_%.3f_%.3f_%d_%d_%d.npz" % (L, eta, eps, t_win0, seed,
                                                   theta0)
        EA_OP = np.load(fin)
        serials = EA_OP["time_serials"]
        EA_OP_arr[i] = np.mean(serials[-50:])
    ax.plot(eps_arr, EA_OP_arr, "-o")
    ax.set_xlabel(r"$\epsilon$", fontsize="large")
    ax.set_ylabel(r"$Q_{\rm EA}$", fontsize="large", color="tab:blue")

    if show_phi:
        phi_arr = np.array([
            decode.cal_order_para(L, eta, eps, seed, theta0) for eps in eps_arr
        ])
        ax_new = ax.twinx()
        ax_new.plot(eps_arr, phi_arr, "-s", c="tab:orange")
        ax_new.tick_params(labelcolor="tab:orange")
        ax_new.set_ylabel(r"$\phi$", color="tab:orange")
        ax.tick_params(labelcolor="tab:blue")
    if flag_show:
        plt.show()
        plt.close()


def plot_EA_OP_vs_t(L, eta, seed, theta0, t_win0, ax=None):
    eps_arr = [0.001, 0.005, 0.01, 0.04, 0.045, 0.05, 0.055, 0.06, 0.1, 0.15]
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    else:
        flag_show = False
    for i, eps in enumerate(eps_arr):
        fin = "EA_OP/%d_%.3f_%.3f_%d_%d_%d.npz" % (L, eta, eps, t_win0, seed,
                                                   theta0)
        EA_OP = np.load(fin)
        serials = EA_OP["time_serials"]
        t = np.arange(serials.size) * t_win0
        ax.plot(t, serials, label="%.3f" % eps)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(0.01, 1)
    ax.set_xlabel(r"$T$", fontsize="large")
    ax.set_ylabel(r"$Q(T)$", fontsize="large")
    ax.legend(title=r"$\epsilon=$")
    if flag_show:
        plt.show()
        plt.close()


def plot_EA_OP_vs_t_eps(L, eta, seed, theta0, t_win0):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    plot_EA_OP_vs_t(L, eta, seed, theta0, t_win0, ax1)
    plot_EA_OP_vs_eps(L, eta, seed, theta0, t_win0, ax2, show_phi=True)
    plt.tight_layout(rect=[-0.015, -0.04, 1.015, 0.98])

    ax3 = fig.add_axes([.74, .5, .18, .4])
    plot_EA_OP_vs_eps(L, eta, seed, theta0, t_win0, ax3)
    ax3.set_ylim(0.005, 0.02)
    ax3.set_xlim(0.05, 0.3)
    fig.suptitle(r"RT: $L=%d, \eta=%g$" % (L, eta), y=0.995, fontsize="large")
    plt.show()
    plt.close()


def plot_EA_OP_2d(L, eta, eps, seed, theta0, t_win0):
    fin = "EA_OP/%d_%.3f_%.3f_%d_%d_%d.npz" % (L, eta, eps, t_win0, seed,
                                               theta0)
    EA_OP = np.load(fin)
    data = EA_OP["last_frame"]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))

    im1 = ax1.imshow(data, origin="lower", extent=(0, L, 0, L))

    from defect import get_long_time_ave_defect
    theta, defects = get_long_time_ave_defect(eps, L, eta, theta0, seed)
    ax1.plot(defects[1]["x"], defects[1]["y"], "wo", fillstyle="none")
    ax1.plot(defects[-1]["x"], defects[-1]["y"], "rx")
    plt.colorbar(im1, ax=ax1, orientation="horizontal")

    fin = "time_ave/RT_ave_%d_%.3f_%.3f_%d_%d.npz" % (L, eta, eps, seed,
                                                      theta0)
    data = np.load(fin)
    vx_m = data["vx_m"]
    vy_m = data["vy_m"]
    module = np.sqrt(vx_m**2 + vy_m**2)
    im2 = ax2.imshow(module, origin="lower", extent=(0, L, 0, L))

    ax1.set_title(
        r"(a) $\langle \overline{{\bf M}}({\bf r}, t)\cdot \overline{{\bf M}}({\bf r}, t+T)\rangle_t$",
        fontsize="x-large")
    ax2.set_title("(b) module of time-averaged momentum fields",
                  fontsize="x-large")

    plt.colorbar(im2, ax=ax2, orientation="horizontal")
    plt.suptitle(r"RT: $L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps),
                 y=0.995,
                 fontsize="xx-large")
    plt.tight_layout(rect=[0, -0.06, 1.04, 0.94])
    plt.show()
    plt.close()


def plot_4_replicas(L, seed, eta, eps):
    from defect import get_long_time_ave_defect
    theta0_arr = [0, 90, 180, 270]
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        theta, defects = get_long_time_ave_defect(eps,
                                                  L,
                                                  eta,
                                                  theta0_arr[i],
                                                  seed,
                                                  sigma=1)
        theta[theta < 0] += np.pi * 2
        ax.imshow(theta, cmap="hsv", origin="lower", extent=(0, L, 0, L))
        ax.plot(defects[1]["x"],
                defects[1]["y"],
                "ko",
                fillstyle="none",
                mew=2)
        ax.plot(defects[-1]["x"],
                defects[-1]["y"],
                "wo",
                fillstyle="none",
                mew=2)
        ax.set_title(r"$n_+=%d, n_-=%d$" %
                     (defects[1]["x"].size, defects[-1]["x"].size))

    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 0.98])
    plt.suptitle(r"$L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps),
                 fontsize="xx-large",
                 y=0.998)
    plt.show()
    plt.close()


def get_colobar_extend(vmin, vmax):
    if vmin is None or vmin == 0.:
        if vmax is None:
            ext = "neither"
        else:
            ext = "max"
    else:
        if vmax is None:
            ext = "min"
        else:
            ext = "both"
    return ext


def varies_t_win(L, eta, eps, seed, theta0, show_defects=True):
    import defect
    fin = "time_ave_200/RT_ave_%d_%g_%g_200_%d_%d.bin" % (L, eta, eps, seed,
                                                          theta0)
    # frame_beg = 500
    # t_beg = 500 * 200 + 300000
    frames = decode.read_time_ave_bin(fin, beg=4000)
    count = 0
    lbox = 4
    rho, vx, vy = np.zeros((3, L // lbox, L // lbox))
    for i, frame in enumerate(frames):
        if i >= 2000:
            break
        rho_t, vx_t, vy_t = frame
        count += 1
        vx += vx_t
        vy += vy_t
        rho += rho_t
        vx_s = gaussian_filter(vx, 1, mode="wrap")
        vy_s = gaussian_filter(vy, 1, mode="wrap")
        theta = np.arctan2(vy_s, vx_s)
        theta[theta < 0] += np.pi * 2
        module = np.sqrt(vx_s**2 + vy_s**2) / count

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))
        ext = [0, L, 0, L]
        im1 = ax1.imshow(rho / count, origin="lower", extent=ext)
        im2 = ax2.imshow(module, origin="lower", extent=ext)
        im3 = ax3.imshow(theta, origin="lower", extent=ext, cmap="hsv")

        plt.colorbar(im1, ax=ax1, orientation="horizontal")
        plt.colorbar(im2, ax=ax2, orientation="horizontal")
        plt.colorbar(im3, ax=ax3, orientation="horizontal")

        if show_defects:
            defects = defect.cal_defects(theta, lbox)
            defect.plot_defects(ax1, defects, 2, 1)
            defect.plot_defects(ax2, defects, 2, 1)
            defect.plot_defects(ax3, defects, 2, 1)

        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) module of momentum")
        ax3.set_title(r"(c) orientation of momentum")
        plt.tight_layout(rect=[-0.015, -0.08, 1.01, 0.97])
        title = r"$L=%d,\eta=%g,\epsilon=%g,\Delta t=%d,n_+=%d,n_+=%d$" % (
            L, eta, eps, count * 200, defects[1]["x"].size,
            defects[-1]["x"].size)
        plt.suptitle(title, y=0.995)
        # plt.show()
        plt.savefig("D:/data/tmp2/dt=%04d.png" % count)
        plt.close()


def varies_tc(L, eta, eps, seed, theta0):
    import defect
    fin = "time_ave_200/RT_ave_%d_%g_%g_200_%d_%d.bin" % (L, eta, eps, seed,
                                                          theta0)
    frame_beg = 3000
    dn = 100
    frames = decode.read_time_ave_bin(fin, beg=frame_beg - dn // 2)
    lbox = 4
    rho, vx, vy = np.zeros((3, dn, L // lbox, L // lbox), np.float32)
    cur_pos = 0
    for i, frame in enumerate(frames):
        t = (frame_beg - dn + i) * 200 + 300000
        if cur_pos >= dn:
            cur_pos -= dn
        rho[cur_pos], vx[cur_pos], vy[cur_pos] = frame
        cur_pos += 1
        if i >= dn - 1:
            rho_m = np.mean(rho, axis=0)
            vx_m = np.mean(vx, axis=0)
            vy_m = np.mean(vy, axis=0)
            vx_m = gaussian_filter(vx_m, 1, mode="wrap")
            vy_m = gaussian_filter(vy_m, 1, mode="wrap")
            theta = np.arctan2(vy_m, vx_m)
            theta[theta < 0] += np.pi * 2
            module = np.sqrt(vx_m**2 + vy_m**2)
            defects = defect.cal_defects(theta, lbox)

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))
            ext = [0, L, 0, L]
            vmin1, vmax1 = None, None
            vmin2, vmax2 = None, None
            im1 = ax1.imshow(rho_m,
                             origin="lower",
                             extent=ext,
                             vmin=vmin1,
                             vmax=vmax1)
            im2 = ax2.imshow(module,
                             origin="lower",
                             extent=ext,
                             vmin=vmin2,
                             vmax=vmax2)
            im3 = ax3.imshow(theta, origin="lower", extent=ext, cmap="hsv")

            ext1 = get_colobar_extend(vmin1, vmax1)
            ext2 = get_colobar_extend(vmin2, vmax2)
            plt.colorbar(im1, ax=ax1, orientation="horizontal", extend=ext1)
            plt.colorbar(im2, ax=ax2, orientation="horizontal", extend=ext2)
            plt.colorbar(im3, ax=ax3, orientation="horizontal")

            defect.plot_defects(ax1, defects, 2, 1)
            defect.plot_defects(ax2, defects, 2, 1)
            defect.plot_defects(ax3, defects, 2, 1)

            ax1.set_title(r"(a) density")
            ax2.set_title(r"(b) module of momentum")
            ax3.set_title(r"(c) orientation of momentum")
            plt.tight_layout(rect=[-0.015, -0.08, 1.01, 0.97])
            title = r"$L=%d,\eta=%g,\epsilon=%g,\Delta t=%d,t=%d,n_+=%d,n_+=%d$" % (
                L, eta, eps, dn * 200, t, defects[1]["x"].size,
                defects[-1]["x"].size)
            plt.suptitle(title, y=0.995)
            # plt.show()
            count = i - dn + 1
            plt.savefig(r"D:/data/tmp/t=%04d.png" % (count))
            plt.close()
            if count >= 200:
                break


def plot_momentum_module_PDF(L=512, eta=0.18, seed=30370000):
    os.chdir("E:/data/random_torque/defects/L=%d_seed=%d/time_ave" % (L, seed))

    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(8, 6),
                             constrained_layout=True)
    axes = axes.flatten()
    titles = [
        r"(a) $\epsilon=%g$", r"(b) $\epsilon=%g$", r"(c) $\epsilon=%g$",
        r"(d) $\epsilon=%g$"
    ]
    eps_arr = np.array([0.06, 0.1, 0.2, 0.3])
    for i, eps in enumerate(eps_arr):
        f1 = "RT_ave_%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed, 100)
        f2 = "RT_ave_%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed, 60)

        data1 = np.load(f1)
        vx, vy = data1["vx_m"], data1["vy_m"]
        module1 = np.sqrt(vx**2 + vy**2)

        data2 = np.load(f2)
        vx, vy = data2["vx_m"], data2["vy_m"]
        module2 = np.sqrt(vx**2 + vy**2)
        if eps == 0.3:
            bins = 100
        else:
            bins = 25
        hist1, bin_edges1 = np.histogram(module1, density=True, bins=bins)
        hist2, bin_edges2 = np.histogram(module2,
                                         density=True,
                                         bins=bin_edges1)

        x1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
        x2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
        axes[i].plot(x1, hist1, "o", fillstyle="none")
        axes[i].plot(x2, hist2, "s", fillstyle="none")
        axes[i].set_yscale("log")
        axes[i].set_xscale("log")
        axes[i].set_title(titles[i] % eps)
    plt.suptitle("PDF for module of long-time averaged momentum fields")
    plt.show()
    plt.close()


if __name__ == "__main__":
    L = 512
    eta = 0.45
    eps = 0.0
    seed = 30370000
    theta0 = 0
    t_win0 = 2000
    os.chdir("E:/data/random_torque/defects/L=%d_seed=%d" % (L, seed))
    # # plot_time_ave_density(L, eta, eps, seed, theta0)

    plot_time_ave_density_momentum(L, eta, eps, seed, theta0)
    # get_all_time_ave(t_win0)

    # cal_all_EA_OP(L, eta, seed, theta0, t_win0)
    # plot_EA_OP_vs_eps(L, eta, seed, theta0, t_win0)
    # plot_EA_OP_vs_t(L, eta, seed, theta0, t_win0)
    # plot_EA_OP_vs_t_eps(L, eta, seed, theta0, t_win0)
    # plot_EA_OP_2d(L, eta, eps, seed, theta0, t_win0)
    # plot_4_replicas(L, seed, eta, eps)

    # varies_t_win(L, eta, eps, seed, theta0)
    # varies_tc(L, eta, eps, seed, theta0)
    # plot_momentum_module_PDF()
