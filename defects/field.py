import os
import numpy as np
import matplotlib.pyplot as plt
from decode import read_field, get_nframe
import time_corr as tc
import glob
# import platform
# from scipy.ndimage import gaussian_filter


def get_para(fin):
    s = os.path.basename(fin).rstrip(".bin").split("_")
    L = int(s[2])
    eta = float(s[3])
    eps = float(s[4])
    t_win0 = int(s[5])
    seed = int(s[6])
    theta0 = int(s[7])
    return L, eta, eps, t_win0, seed, theta0


def get_momentum_serials(L=512,
                         eta=0.18,
                         eps=0.05,
                         seed=30370000,
                         theta0=0,
                         t_win=200,
                         t_win0=200,
                         frame_beg=500,
                         fin=None,
                         lbox=4):
    if fin is None:
        fin = "field_%d/RT_feild_%d_%.3f_%.3f_%d_%d_%03d.bin" % (
            t_win0, L, eta, eps, t_win0, seed, theta0)
    else:
        L, eta, eps, t_win0, seed, theta0 = get_para(fin)
    max_frame = get_nframe(fin, lbox=lbox)
    print("max frame =", max_frame)
    frame_sep = t_win // t_win0
    n_frame = (max_frame - frame_beg + 1) // frame_sep
    print("n_frame =", n_frame, "sep =", frame_sep)
    vx, vy = np.zeros((2, n_frame, L // lbox, L // lbox), np.float32)
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


def show_fields(fin):
    L, eta, eps, t_win0, seed, theta0 = get_para(fin)
    frames = read_field(fin)
    for i, (rho, vx, vy) in enumerate(frames):
        # vx = gaussian_filter(vx, sigma=1)
        # vy = gaussian_filter(vy, sigma=1)
        theta = np.arctan2(vy, vx)
        theta[theta < 0] += np.pi * 2
        module = np.sqrt(vx**2 + vy**2)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))
        ext = [0, L, 0, L]
        vmin1, vmax1 = None, None
        vmin2, vmax2 = None, None
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
        im3 = ax3.imshow(theta, origin="lower", extent=ext, cmap="hsv")

        ext1 = get_colobar_extend(vmin1, vmax1)
        ext2 = get_colobar_extend(vmin2, vmax2)
        plt.colorbar(im1, ax=ax1, orientation="horizontal", extend=ext1)
        plt.colorbar(im2, ax=ax2, orientation="horizontal", extend=ext2)
        plt.colorbar(im3, ax=ax3, orientation="horizontal")

        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) module of momentum")
        ax3.set_title(r"(c) orientation of momentum")
        plt.tight_layout(rect=[-0.015, -0.08, 1.01, 0.97])
        t = (i + 1) * t_win0 + 300000
        title = r"$L=%d,\eta=%g,\epsilon=%g,t=%d$" % (L, eta, eps, t)
        title = "instantaneous fields: " + title
        plt.suptitle(title, y=0.995, fontsize="x-large")

        # plt.show()
        plt.savefig(r"D:/data/tmp2/t=%04d.png" % i)
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
    os.chdir("E:/data/random_torque/defects/samples")
    cal_EA_order_para(1000, 1000)
    # varied_L()

    # eta = 0.45
    # eps = 0.1
    # seed = 30370000
    # twin0 = 200
    # os.chdir("E:/data/random_torque/defects/L=512_seed=%d/field_%d" %
    #          (seed, twin0))
    # fin = "RT_field_512_%.3f_%.3f_%d_%d_000.bin" % (eta, eps, twin0, seed)
    # show_fields(fin)
