import os
import numpy as np
import matplotlib.pyplot as plt
from decode import read_field, get_nframe
import time_corr as tc
import glob
import sys
sys.path.append("..")
try:
    from snap.plot_snap import map_v_to_rgb, add_colorbar
except ModuleNotFoundError:
    print("failed to import map_v_to_rgb, add_colorbar")
    exit()


def get_para(fin):
    s = os.path.basename(fin).rstrip(".bin").split("_")
    para = {}
    if len(s) == 8:
        para["Lx"] = int(s[2])
        para["Ly"] = para["Lx"]
        para["eta"] = float(s[3])
        para["eps"] = float(s[4])
        para["t_win0"] = int(s[5])
        para["seed"] = int(s[6])
        para["theta0"] = int(s[7])
    else:
        para["Lx"] = int(s[2])
        para["Ly"] = int(s[3])
        para["eta"] = float(s[4])
        para["eps"] = float(s[5])
        para["t_win0"] = int(s[6])
        para["seed"] = int(s[7])
        para["theta0"] = int(s[8])
    return para


def get_momentum_serials(para, t_win=200, frame_beg=500, fin=None, lbox=4):
    if fin is None:
        fin = "field_%d/RT_feild_%d_%.3f_%.3f_%d_%d_%03d.bin" % (
            para["t_win0"], para["Lx"], para["eta"], para["eps"],
            para["t_win0"], para["seed"], para["theta0"])
    else:
        para = get_para(fin)
    para["t_win"] = t_win
    max_frame = get_nframe(fin, lbox=lbox)
    print("max frame =", max_frame)
    frame_sep = t_win // para["t_win0"]
    n_frame = (max_frame - frame_beg + 1) // frame_sep
    print("n_frame =", n_frame, "sep =", frame_sep)
    ncols, nrows = para["Lx"] // lbox, para["Ly"] // lbox
    vx, vy = np.zeros((2, n_frame, nrows, ncols), np.float32)
    frames = read_field(fin, frame_beg, None, frame_sep, lbox)
    for i, (rho_t, vx_t, vy_t) in enumerate(frames):
        vx[i] = vx_t
        vy[i] = vy_t
    return vx, vy


def cal_EA_order_para(t_win0=200, t_win=200):
    if not os.path.exists("EA_OP"):
        os.mkdir("EA_OP")
    files = glob.glob("RT_field*.bin")

    for fin in files:
        L, eta, eps, t_win0, seed, theta0 = get_para(fin)
        fout = "EA_OP/%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (L, eta, eps, t_win0,
                                                         t_win, seed, theta0)
        if os.path.exists(
                fout) and os.path.getmtime(fout) > os.path.getmtime(fin):
            print(fout, "is up to date.")
            continue
        else:
            print("update", fout)
        vx, vy = get_momentum_serials(fin=fin,
                                      lbox=4,
                                      t_win=t_win,
                                      frame_beg=0,
                                      t_win0=t_win0)
        EA_OP_t, EA_OP_r = tc.cal_EA_OP(vx, vy)
        np.savez_compressed(fout, time_serials=EA_OP_t, last_frame=EA_OP_t)


def varied_L(L, eta, seed):
    os.chdir("E:/data/random_torque/defects")
    eps_arr1 = np.array([
        0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.1,
        0.15, 0.2, 0.25, 0.3
    ])
    L = 512
    EA_OP_arr1 = np.zeros_like(eps_arr1)
    for i, eps in enumerate(eps_arr1):
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, 200)
        basename = "%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (L, eta, eps, 200, 200,
                                                       seed, 0)
        fin = folder + os.path.sep + basename
        data1 = np.load(fin)
        EA_OP = data1["time_serials"]
        EA_OP_arr1[i] = EA_OP[-1]
        t = np.arange(EA_OP.size) * 200
        plt.loglog(t, EA_OP, label="%.3f" % eps)
    plt.legend()
    plt.show()
    plt.close()

    L = 256
    eps_arr2 = np.array([0.055, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25])
    EA_OP_arr2 = np.zeros_like(eps_arr2)
    for i, eps in enumerate(eps_arr2):
        folder = "L=%d_seed=%d/field_%d/EA_OP" % (L, seed, 200)
        basename = "%d_%.3f_%.3f_%d_%d_%d_%03d.npz" % (L, eta, eps, 200, 200,
                                                       seed, 0)
        fin = folder + os.path.sep + basename
        data1 = np.load(fin)
        EA_OP = data1["time_serials"]
        EA_OP_arr2[i] = EA_OP[-1]
        t = np.arange(EA_OP.size) * 200
        plt.loglog(t, EA_OP, label="%.3f" % eps)
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(eps_arr1, EA_OP_arr1, "-o")
    plt.plot(eps_arr2, EA_OP_arr2, "-s")
    # plt.yscale("log")
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


def show_fields(fin, t_beg=300000):
    para = get_para(fin)
    frames = read_field(fin)
    for i, (rho, vx, vy) in enumerate(frames):
        # vx = gaussian_filter(vx, sigma=1)
        # vy = gaussian_filter(vy, sigma=1)
        theta = np.arctan2(vy, vx)
        theta[theta < 0] += np.pi * 2
        module = np.sqrt(vx**2 + vy**2)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))
        ext = [0, L, 0, L]
        vmin1, vmax1 = None, 6
        vmin2, vmax2 = None, 10
        im1 = ax1.imshow(rho,
                         origin="lower",
                         extent=ext,
                         vmin=vmin1,
                         vmax=vmax1)
        im2 = ax2.imshow(module,
                         origin="lower",
                         extent=ext,
                         vmin=vmin2,
                         vmax=vmax2)
        im3 = ax3.imshow(theta,
                         origin="lower",
                         extent=ext,
                         cmap="hsv",
                         vmin=0,
                         vmax=np.pi * 2)

        ext1 = get_colobar_extend(vmin1, vmax1)
        ext2 = get_colobar_extend(vmin2, vmax2)
        plt.colorbar(im1, ax=ax1, orientation="horizontal", extend=ext1)
        plt.colorbar(im2, ax=ax2, orientation="horizontal", extend=ext2)
        plt.colorbar(im3, ax=ax3, orientation="horizontal")

        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) module of momentum")
        ax3.set_title(r"(c) orientation of momentum")
        plt.tight_layout(rect=[-0.015, -0.08, 1.01, 0.97])
        para["t"] = (i + 1) * para["t_win0"] + t_beg
        title = "$L={Lx},\\eta={eta:g},\epsilon={eps:g},t={t}$".format(**para)
        title = "instantaneous fields: " + title
        plt.suptitle(title, y=0.995, fontsize="x-large")

        # plt.show()
        plt.savefig(r"D:/data/tmp2/t=%04d.png" % i)
        plt.close()


def show_fields2(fin, t_beg=300000, lbox=4, flag_save=False, disorder="RS"):
    para = get_para(fin)
    frames = read_field(fin, lbox=lbox)

    if flag_save:
        folder = "D:/data/tmp/%s" % (os.path.basename(fin).rstrip(".bin"))
        if not os.path.exists(folder):
            os.mkdir(folder)
    for i, (rho, vx, vy) in enumerate(frames):
        if flag_save:
            outfile = "%s/t=%04d.png" % (folder, i)
            if os.path.exists(outfile):
                print(f"skip frame {i}")
                continue
        theta = np.arctan2(vy, vx)
        theta[theta < 0] += np.pi * 2
        theta *= 180 / np.pi
        module = np.sqrt(vx**2 + vy**2)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
        box = [0, para["Lx"], 0, para["Ly"]]
        vmin1, vmax1 = None, 6
        vmin2, vmax2 = 0, 4
        im1 = ax1.imshow(rho,
                         origin="lower",
                         extent=box,
                         vmin=vmin1,
                         vmax=vmax1)
        RGB = map_v_to_rgb(theta, module, m_max=vmax2)
        ax2.imshow(RGB, extent=box, origin="lower")
        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) momentum")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        bbox1 = ax1.get_position().get_points().flatten()
        bbox2 = ax2.get_position().get_points().flatten()
        fig.subplots_adjust(bottom=0.24)
        bbox1[1], bbox1[3] = 0.14, 0.04
        bbox1[2] = bbox1[2] - bbox1[0] - 0.03
        bbox2[1], bbox2[3] = 0.08, 0.14
        bbox2[2] = bbox2[2] - bbox2[0]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)
        ext1 = get_colobar_extend(vmin1, vmax1)
        cb1 = fig.colorbar(im1,
                           ax=ax1,
                           cax=cb_ax1,
                           orientation="horizontal",
                           extend=ext1)
        cb1.set_label(r"density $\rho$", fontsize="x-large")
        add_colorbar(cb_ax2, vmin2, vmax2, 0, 360)

        t = (i + 1) * para["t_win0"] + t_beg
        title = r"%s: $L=%d,\eta=%g,\epsilon=%g,\theta_0=%d,t=%d$" % (
            disorder, para["Lx"], para["eta"], para["eps"], para["theta0"], t)
        plt.suptitle(title, y=0.995, fontsize="x-large")

        if flag_save:
            plt.savefig(outfile)
            print(f"save frame {i}")
        else:
            plt.show()
        plt.close()


def show_fields_rect(fin, t_beg=0, lbox=8, flag_save=False, disorder="RS"):
    para = get_para(fin)
    frames = read_field(fin, lbox=lbox)

    for i, (rho, vx, vy) in enumerate(frames):
        theta = np.arctan2(vy, vx)
        theta[theta < 0] += np.pi * 2
        theta *= 180 / np.pi
        module = np.sqrt(vx**2 + vy**2)
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(19, 4), sharex=True)
        box = [0, para["Lx"], 0, para["Ly"]]
        vmin1, vmax1 = None, 6
        vmax2 = 4
        ax1.imshow(rho, origin="lower", extent=box, vmin=vmin1, vmax=vmax1)
        RGB = map_v_to_rgb(theta, module, m_max=vmax2)
        ax2.imshow(RGB, extent=box, origin="lower")
        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) momentum")
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        t = (i + 1) * para["t_win0"] + t_beg
        title = r"%s: $L_x=%d,L_y=%d,\eta=%g,\epsilon=%g,\theta_0=%d,t=%d$" % (
            disorder, para["Lx"], para["Ly"], para["eta"], para["eps"],
            para["theta0"], t)
        plt.tight_layout()
        plt.suptitle(title, y=0.995, fontsize="x-large")
        if flag_save:
            plt.savefig(r"D:/data/tmp/t=%04d.png" % i)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    # L = 256
    # seed = 30370002
    # t_win0 = 200
    # theta0 = 0
    # if platform.system() == 'Windows':
    #     os.chdir(r"E:/data/random_torque/defects/L=%d_seed=%d/field_%d" %
    #              (L, seed, t_win0))
    # else:
    #     os.chdir("data")
    # os.chdir("E:/data/random_torque/defects/samples")
    # cal_EA_order_para(1000, 1000)
    # varied_L()

    # eta = 0.45
    # eps = 0.1
    # seed = 30370000
    # twin0 = 200
    # os.chdir("E:/data/random_torque/defects/L=512_seed=%d/field_%d" %
    #          (seed, twin0))
    # fin = "RT_field_512_%.3f_%.3f_%d_%d_000.bin" % (eta, eps, twin0, seed)
    # show_fields(fin)

    L = 8192
    eta = 0.18
    eps = 0.035
    seed = 20200712
    twin0 = 500
    theta0 = 0
    os.chdir("E:/data/random_torque/replica2/L=%d" % L)
    fin = "RT_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (L, eta, eps, twin0, seed,
                                                    theta0)
    show_fields2(fin, t_beg=0, lbox=8, flag_save=True)

    # Lx = 16384
    # Ly = 1024
    # eta = 0.18
    # eps = 0.035
    # seed = 20200712
    # twin0 = 500
    # theta0 = 180
    # os.chdir("E:/data/random_torque/replica2/Rect_wall_y/")
    # fin = "RT_field_%d_%d_%.3f_%.3f_%d_%d_%03d.bin" % (Lx, Ly, eta, eps, twin0,
    #                                                    seed, theta0)
    # show_fields_rect(fin, flag_save=True)

    # L = 2048
    # eta = 0.18
    # eps = 0.09
    # seed = 20200712
    # twin0 = 500
    # theta0 = 90
    # os.chdir("E:/data/random_field/normalize_new/replica")
    # fin = "RF_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (L, eta, eps, twin0, seed,
    #                                                 theta0)
    # show_fields2(fin, t_beg=0, lbox=8, flag_save=True, disorder="RF")

    # L = 1024
    # eta = 0
    # eps = 0.2
    # seed = 30370000
    # twin0 = 1000
    # theta0 = 0
    # os.chdir("E:/data/random_potential/replicas/field")
    # fin = "RP_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (L, eta, eps, twin0, seed,
    #                                                 theta0)
    # show_fields2(fin, t_beg=0, lbox=4, flag_save=True, disorder="RC")
