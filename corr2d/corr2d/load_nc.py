"""
Load correlation functions and snapshots from netcdf4 files.

@author skyline-nju
@date 2018-05-08
"""
from netCDF4 import Dataset
import os
import numpy as np
import platform
import matplotlib
if platform.system != "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
else:
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb


class Corr2d:
    def __init__(self, infile):
        self.rootgrp = Dataset(infile, "r", format="NETCDF4")

    def gene_frame(self, beg=0, end=None, sep=1):
        nframes = len(self.rootgrp.dimensions["frame"])
        if end is None:
            end = nframes
        for i in range(beg, end, sep):
            t = self.rootgrp.variables["time"][i:i + 1][0]
            vm = self.rootgrp.variables["mean_velocity"][i:i + 1][0]
            c_rho = np.array(self.rootgrp.variables["C_rho"][i:i + 1][0])
            c_v = np.array(self.rootgrp.variables["C_v"][i:i + 1][0])
            yield [t, vm, c_rho, c_v]


class CGSnap:
    def __init__(self, infile):
        self.rootgrp = Dataset(infile, "r", format="NETCDF4")
        self.infile = infile
        s = infile.replace(".nc", "").split("_")
        self.eta = float(s[1])
        self.eps = float(s[2])
        self.rho0 = float(s[3])
        self.Lx = float(s[4])
        self.Ly = float(s[5])
        self.seed = int(s[6])
        self.ncols = len(self.rootgrp.dimensions["ncols"])
        self.nrows = len(self.rootgrp.dimensions["nrows"])

    def gene_frame(self, beg=0, end=None, sep=1):
        nframes = len(self.rootgrp.dimensions["frame"])
        if end is None:
            end = nframes
        for i in range(beg, end, sep):
            t = self.rootgrp.variables["time"][i:i + 1][0]
            num = self.rootgrp.variables["num"][i:i + 1][0]
            vx = self.rootgrp.variables["vx"][i:i + 1][0]
            vy = self.rootgrp.variables["vy"][i:i + 1][0]
            yield [t, num, vx, vy]

    def show(self, savefig=False, v_normed=False):
        if savefig or platform.system != "Windows":
            savefig = True
            fig_dir = self.infile.replace(".nc", "")
            os.mkdir(fig_dir)
        else:
            fig_dir = r"./"

        domain = [0, self.Lx, 0, self.Ly]
        rho_level = np.linspace(0, 4, 9)
        dx = self.Lx / self.ncols
        dy = self.Ly / self.nrows
        dA = dx * dy
        x = np.linspace(dx / 2, self.Lx - dx / 2, self.ncols)
        y = np.linspace(dy / 2, self.Ly - dy / 2, self.nrows)
        if self.Lx == self.Ly:
            title_pat = r"$\eta=%g, \epsilon=%g, \rho_0=%g, L=%g," + \
                r"{\rm, seed}=%g, t=%d$"
        frames = self.gene_frame()
        for i, frame in enumerate(frames):
            t, num, vx, vy = frame
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
            rho = num / dA
            contour = ax1.contourf(x, y, rho, rho_level, extend="max")
            ax1.axis("scaled")
            ax1.axis(domain)

            v_orient = np.arctan2(vy, vx) / np.pi * 180
            v_orient[v_orient < 0] += 360
            v_module = np.sqrt(vx**2 + vy**2)
            if v_normed:
                mask = num != 0
                v_module[mask] /= num[mask]
            RGB = map_v_to_rgb(v_orient, v_module, m_max=4)
            ax2.imshow(RGB, extent=domain, origin="lower")
            ax2.axis('scaled')
            ax2.axis(domain)
            # n_tot = np.sum(num)
            # vxm = np.sum(vx) / n_tot
            # vym = np.sum(vy) / n_tot
            # phi = np.sqrt(vxm**2 + vym**2)

            plt.suptitle(
                title_pat % (self.eta, self.eps, 1, self.Lx, self.seed, t),
                fontsize="xx-large",
                y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            bbox1 = ax1.get_position().get_points().flatten()
            bbox2 = ax2.get_position().get_points().flatten()
            fig.subplots_adjust(bottom=0.24)
            bbox1[1], bbox1[3] = 0.14, 0.04
            bbox1[2] = bbox1[2] - bbox1[0] - 0.03
            bbox2[1], bbox2[3] = 0.08, 0.14
            bbox2[2] = bbox2[2] - bbox2[0]
            cb_ax1 = fig.add_axes(bbox1)
            cb_ax2 = fig.add_axes(bbox2)
            cb1 = fig.colorbar(contour, cax=cb_ax1, orientation="horizontal")
            cb1.set_label(r"density $\rho$", fontsize="x-large")
            add_colorbar(cb_ax2, 0, 1, 0, 360)
            if savefig:
                plt.savefig(fig_dir + os.path.sep + "%04d.png" % i)
            else:
                plt.show()
            plt.close()


def map_v_to_rgb(theta, module, m_max=None):
    """
    Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta: array_like
        Orietation of velocity field.
    module: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $\rho |v|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        ax.set_xlabel(r"module $\rho |v|$", fontsize="large")


def cal_corr2d(num, vx, vy, cell_area):
    def auto_corr2d(f):
        F = np.fft.rfft2(f)
        h = np.fft.irfft2(F * F.conj())
        return h

    rho = num / cell_area
    corr_rho = auto_corr2d(rho)
    corr_v = (auto_corr2d(vx) + auto_corr2d(vy)) / corr_rho
    corr_rho /= num.size
    return corr_rho, corr_v


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    os.chdir(r"D:\tmp")
    # f = Corr2d(r"cr_0.18_0_1_512_512_2.nc")
    # frames = f.gene_frame()
    # for frame in frames:
    #     t, vm, c_rho, c_v = frame
    #     # t, num, vx, vy = frame
    #     print(t, c_v[0, 250])
    #     # plt.imshow(c_v, origin="lower")
    #     # plt.show()
    #     # plt.close()
    f2 = CGSnap(r"cg_0.18_0_1_512_512_2.nc")
    f2.show(True)
    # frames = f2.gene_frame()
    # for frame in frames:
    #     t, num, vx, vy = frame
    #     corr_rho, corr_v = cal_corr2d(num, vx, vy, 1)
    #     print(t, corr_v[0, 250])
