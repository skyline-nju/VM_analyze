import numpy as np
import os
import matplotlib.pyplot as plt
from decode import read_field
from field import get_para


def cal_corr2d(f):
    F = np.fft.rfft2(f)
    h = np.fft.irfft2(F * F.conj()) / f.size
    return h


def time_ave_corr2d(fin, lbox, start=0, sep=1, save=False, fout=None):
    frames = read_field(fin, lbox=lbox, beg=start, sep=sep)
    for i, (rho, vx, vy) in enumerate(frames):
        if i == 0:
            C_rho = cal_corr2d(rho)
            C_J = cal_corr2d(vx) + cal_corr2d(vy)
        else:
            C_rho += cal_corr2d(rho)
            C_J += cal_corr2d(vx) + cal_corr2d(vy)
    C_rho /= (i + 1)
    C_J /= (i + 1)
    C_rho = np.fft.fftshift(C_rho)
    C_J = np.fft.fftshift(C_J)
    if not save:
        return C_rho, C_J
    else:
        if not os.path.exists("spatial_corr"):
            os.mkdir("spatial_corr")
        if fout is None:
            para = get_para(fin)
            fout = "spatial_corr/%d_%.3f_%.3f_%d_%03d.npz" % (
                para["Lx"], para["eta"], para["eps"], para["seed"],
                para["theta0"])
        np.savez(fout, C_rho=C_rho, C_J=C_J)


def spherical_average(corr, L, smoothed=False):
    """
        Average correlation function spherically.

        Parameters:
        --------
        corr : array_like
            Input correlation function with dimension two.
        L : int
            Size of the system.
        smoothed : bool, optional
            If true, smooth the averaged correlation function.

        Returns:
        --------
        r : array_like
            The array of radius.
        corr_r : array_like
            Averaged correlation function circularly.
    """
    nRow, nCol = corr.shape
    dict_count = {}
    dict_cr = {}
    rr_max = ((min(nRow, nCol)) // 2)**2
    for row in range(nRow):
        dy = row - nRow // 2
        for col in range(nCol):
            dx = col - nCol // 2
            rr = dx * dx + dy * dy
            if rr < rr_max:
                if rr in dict_count.keys():
                    dict_count[rr] += 1
                    dict_cr[rr] += corr[row, col]
                else:
                    dict_count[rr] = 1
                    dict_cr[rr] = corr[row, col]
    rr_sorted = np.array(sorted(dict_count.keys()))
    r = np.sqrt(rr_sorted) * L / nRow
    corr_r = np.array([dict_cr[key] / dict_count[key] for key in rr_sorted])
    # if smoothed:
    #     r, corr_r = smooth(r, corr_r)
    return r, corr_r


def plot_Cr(L, eta, eps, theta0, seed, ax):
    folder = "D:/data/vm2d/random_torque/spatial_corr/"
    fin = "%s/%d_%.3f_%.3f_%d_%03d.npz" % (folder, L, eta, eps, seed, theta0)
    C_rho = np.load(fin)["C_rho"]
    r, Cr_rho = spherical_average(C_rho, L)
    label = r"$L=%d, \eta=%g, \epsilon=%g$" % (L, eta, eps)
    alpha = 0
    ax.plot(r / L**alpha, Cr_rho, "o", ms=2, label=label)
    ax.set_xscale("log")
    # ax.set_yscale("log")


def cal_spatial_corr(L, eta, eps, theta0, seed, twin0=1000, lbox=4):
    if L == 2048:
        folder_in = "E:/data/random_torque/replica2/L=2048"
        fin = "%s/RT_field_%d_%.3f_%.3f_%d_%03d_0_10000.bin" % (
            folder_in, L, eta, eps, seed, theta0)
        start = 40
        if eps == 0.045:
            start = 60
            print("start frame = %d for eps = %g" % (start, eps))
    else:
        folder_in = "E:/data/random_torque/defects/L=%d_seed=%d/field_%d" % (
            L, seed, twin0)
        fin = "%s/RT_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (
            folder_in, L, eta, eps, twin0, seed, theta0)
        start = 0

    folder_out = "D:/data/VM2d/random_torque/spatial_corr"
    fout = "%s/%d_%.3f_%.3f_%d_%03d.npz" % (folder_out, L, eta, eps, seed,
                                            theta0)
    time_ave_corr2d(fin, lbox, save=True, start=start, fout=fout)


if __name__ == "__main__":
    # # cal_spatial_corr(512, 0.05, 0.06, 0, 30370000, 1000)
    # fig, ax = plt.subplots()

    # plot_Cr(512, 0.18, 0.055, 0, 30370000, ax)
    # plot_Cr(512, 0.05, 0.06, 0, 30370000, ax)
    # plot_Cr(512, 0.18, 0.06, 0, 30370000, ax)
    # plot_Cr(512, 0.45, 0.06, 0, 30370000, ax)
    # plot_Cr(512, 0.45, 0., 0, 30370000, ax)
    # # plot_Cr(2048, 0.18, 0.03, 0, 20200712, ax)
    # # plot_Cr(2048, 0.18, 0.035, 0, 20200712, ax)
    # # plot_Cr(2048, 0.18, 0.04, 0, 20200712, ax)
    # # plot_Cr(2048, 0.18, 0.045, 0, 20200712, ax)
    # # plot_Cr(2048, 0.18, 0.05, 0, 20200712, ax)
    # # plot_Cr(2048, 0.18, 0.055, 0, 20200712, ax)

    # ax.legend(fontsize="large")
    # ax.set_xlabel(r"$r$", fontsize="x-large")
    # ax.set_ylabel(r"$g(r)$", fontsize="x-large")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    L = 8192
    eta = 0.18
    eps = 0.035
    seed = 20200712
    twin0 = 500
    theta0 = 0

    os.chdir("E:/data/random_torque/replica2/L=%d" % (L))
    fin = "RT_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (L, eta, eps, twin0, seed,
                                                    theta0)
    # time_ave_corr2d(fin, 8, 1000, 1, True)

    fin = "spatial_corr/%d_%.3f_%.3f_%d_%03d.npz" % (L, eta, eps, seed, theta0)
    data = np.load(fin)
    C_rho = data["C_rho"]
    C_J = data["C_J"]
    plt.subplot(121)
    plt.imshow(C_rho, origin="lower", cmap="turbo")
    plt.subplot(122)
    plt.imshow(C_J, origin="lower", cmap="turbo")
    plt.show()
    plt.close()

    # r, Cr = spherical_average(C_rho, 4096)
    # plt.plot(r, Cr)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # plt.close()

    r = np.arange(512) * 8
    C1 = C_J[512, 512:]
    C2 = C_J[512:, 512]
    plt.loglog(r, C1)
    plt.loglog(r, C2)
    plt.show()
    plt.close()
