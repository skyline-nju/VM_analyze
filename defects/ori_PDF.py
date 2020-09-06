""" Show PDF of orientation fields. """

import numpy as np
import matplotlib.pyplot as plt
import os

from decode import read_field, read_snap
from field import get_para


def ori_field_PDF(fin, t_beg=0, lbox=8, frame_beg=1000):
    """Show PDf of oritantion field

    Args:
        fin (str): input file recording intant fields
        t_beg (int, optional): t for 0th frame. Defaults to 0.
        lbox (int, optional): liner size of boxes for coarse graining.
        Defaults to 8.
        frame_beg (int, optional): first frame to show. Defaults to 1000.
    """
    para = get_para(fin)
    frames = read_field(fin, lbox=lbox, beg=frame_beg)
    plt.ion()
    for i, (rhox, vx, vy) in enumerate(frames):
        theta = np.arctan2(vy, vx) / np.pi
        module = np.sqrt(vx**2 + vy**2)
        bins = 60
        angle_range = (-1, 1)
        hist1, bin_edges = np.histogram(theta,
                                        bins=bins,
                                        density=True,
                                        range=angle_range)
        hist2, bin_edges = np.histogram(theta,
                                        bins=bins,
                                        density=True,
                                        weights=module,
                                        range=angle_range)
        theta = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(theta, hist1, "-o")
        plt.plot(theta, hist2, "-s")
        t = (i + 1 + frame_beg) * para["t_win0"] + t_beg
        plt.title(r"$t=%d$" % t)
        plt.xlabel(r"$\theta/\pi$")
        plt.draw()
        plt.pause(0.1)
        plt.cla()
    plt.close()


def ori_snap_PDF():
    pass


if __name__ == "__main__":
    L = 8192
    eta = 0.18
    eps = 0.035
    seed = 20200712
    twin0 = 500
    theta0 = 270
    os.chdir("E:/data/random_torque/replica2/L=%d" % L)
    # fin = "RT_field_%d_%.3f_%.3f_%d_%d_%03d.bin" % (L, eta, eps, twin0, seed,
    #                                                 theta0)
    # ori_field_PDF(fin, frame_beg=2000)
    # fin = "snap/s%d.%g.%g.%d.%d.%04d.bin" % (L, eta * 1000, eps * 1000, seed,
    #                                          theta0, 2)
    # x, y, theta = read_snap(fin)
    # bins = 120
    # angle_range = (-1, 1)
    # hist, bin_edges = np.histogram(theta / np.pi,
    #                                bins=bins,
    #                                density=True,
    #                                range=angle_range)
    # x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # plt.plot(x, hist, "-o", fillstyle="none")
    # plt.show()
    # plt.close()

    # np.savez("snap/PDF_%d_%g_%g_%d_%03d_%04d.npz" %
    #          (L, eta, eps, seed, theta0, 2),
    #          theta=x,
    #          pdf=hist)

    for theta0 in [0, 90]:
        data = np.load("snap/PDF_%d_%g_%g_%d_%03d_%04d.npz" %
                       (L, eta, eps, seed, theta0, 2))
        theta = data["theta"]
        pdf = data["pdf"]

        plt.plot(theta,
                 pdf,
                 "-o",
                 fillstyle="none",
                 label=r"$\theta_0=%d$" % theta0)
    plt.legend()
    plt.xlabel(r"$\theta /\pi$")
    plt.ylabel("PDF")
    plt.title(r"RS: $L=%d, \eta=%g, \epsilon=%g, t=%d$" % (L, eta, eps, 8e5))
    plt.show()
