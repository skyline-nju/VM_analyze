"""
Load correlation functions and snapshots from netcdf4 files.

@author skyline-nju
@date 2018-05-08
"""
from netCDF4 import Dataset
import os
import glob
import numpy as np
import platform
import matplotlib
from scipy.interpolate import RectBivariateSpline
if platform.system() != "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
else:
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb


class Corr2d:
    def __init__(self, infile):
        self.rootgrp = Dataset(infile, "r", format="NETCDF4")
        self.eta = getattr(self.rootgrp, "eta")
        self.eps = getattr(self.rootgrp, "epsilon")
        self.ncols = self.rootgrp.dimensions["ncols"].size
        self.nrows = self.rootgrp.dimensions["nrows"].size
        self.Lx = getattr(self.rootgrp, "Lx")
        self.Ly = getattr(self.rootgrp, "Ly")
        self.x = np.linspace(
            -self.Lx / 2, self.Lx / 2, self.ncols, endpoint=False)
        self.y = np.linspace(0, self.Ly / 2, self.nrows, endpoint=False)
        self.r = self.y

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

    def show(self):
        frames = self.gene_frame()
        for i, frame in enumerate(frames):
            t, vm, c_rho, c_v = frame
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7.5))
            im1 = ax1.imshow(c_rho, origin="lower")
            plt.colorbar(mappable=im1, ax=ax1)
            im2 = ax2.imshow(c_v, origin="lower")
            plt.colorbar(mappable=im2, ax=ax2)
            plt.show()
            plt.close()

    def interp(self, frame):
        """ Get instant correlation functions for density and velocity in
            longitudinal and transversal directions by interpolations.
        """
        t, vm, c_rho, c_v = frame
        c_rho_new = np.zeros_like(c_rho)
        c_rho_new[:, 0:self.ncols // 2] = c_rho[:, self.ncols // 2:]
        c_rho_new[:, self.ncols // 2:] = c_rho[:, 0:self.ncols // 2]
        c_v_new = np.zeros_like(c_v)
        c_v_new[:, 0:self.ncols // 2] = c_v[:, self.ncols // 2:]
        c_v_new[:, self.ncols // 2:] = c_v[:, 0:self.ncols // 2]
        f_rho = RectBivariateSpline(self.x, self.y, c_rho_new.T)
        f_v = RectBivariateSpline(self.x, self.y, c_v_new.T)
        theta = np.arctan2(vm[1], vm[0])
        if theta < 0:
            theta += np.pi
        xl = self.r * np.cos(theta)
        yl = self.r * np.sin(theta)
        xt = self.r * np.cos(np.pi - theta)
        yt = self.r * np.sin(np.pi - theta)

        c_rho_l = f_rho(xl, yl, grid=False)
        c_rho_t = f_rho(xt, yt, grid=False)
        c_v_l = f_v(xl, yl, grid=False)
        c_v_t = f_v(xt, yt, grid=False)
        return c_rho_l, c_rho_t, c_v_l, c_v_t

    def show_corr_lon_tra(self, save_data=False):
        """ Show time-averaged correlation function for density and velocity
            in the longitudinal and transversal directions.
        """
        frames = self.gene_frame()
        crho_lon, crho_tra, cv_lon, cv_tra = np.zeros((4, self.r.size))
        count = 0
        for i, frame in enumerate(frames):
            c_rho_l, c_rho_t, c_v_l, c_v_t = self.interp(frame)
            crho_lon += c_rho_l
            crho_tra += c_rho_t
            cv_lon += c_v_l
            cv_tra += c_v_t
            count += 1
        crho_lon /= count
        crho_tra /= count
        cv_lon /= count
        cv_tra /= count
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax1.plot(self.r, crho_lon, "s", label="longitudinal")
        ax1.plot(self.r, crho_tra, "o", label="transversal")
        ax1.set_xlabel(r"$r$")
        ax1.set_ylabel(r"$C_\rho$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax2.loglog(self.r, cv_lon, "s", label="longitudinal")
        ax2.loglog(self.r, cv_tra, "o", label="transversal")
        ax2.set_xlabel(r"$r$")
        ax2.set_ylabel(r"$C_v$")
        ax2.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.suptitle("$\eta=%g,\ \epsilon=%g,\ L=%d$" % (self.eta, self.eps,
                                                         self.Lx))
        plt.show()
        plt.close()

        if save_data:
            with open("cr_lon_tra_%g_%g_%d.dat" % (
                    self.eta, self.eps, self.Lx), "w") as f:
                lines = ""
                for i, r in enumerate(self.r):
                    lines += "%g\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
                        r, crho_lon[i], crho_tra[i], cv_lon[i], cv_tra[i])
                f.write(lines)


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
        self.fig_dir = self.infile.replace(".nc", "")
        self.mp4_file = "%g_%g_%d_%d.mp4" % (self.eta, self.eps,
                                             self.Lx, self.seed)
        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)

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
        if savefig or platform.system() != "Windows":
            savefig = True

        domain = [0, self.Lx, 0, self.Ly]
        rho_level = np.linspace(0, 4, 9)
        dx = self.Lx / self.ncols
        dy = self.Ly / self.nrows
        dA = dx * dy
        x = np.linspace(dx / 2, self.Lx - dx / 2, self.ncols)
        y = np.linspace(dy / 2, self.Ly - dy / 2, self.nrows)
        if self.Lx == self.Ly:
            title_pat = r"$\eta=%g, \epsilon=%g, \rho_0=%g, L=%g," + \
                r"{\rm, seed}=%g, t=%d, \phi=%.4f$"
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
                RGB = map_v_to_rgb(v_orient, v_module)
            else:
                RGB = map_v_to_rgb(v_orient, v_module, m_max=4)
            ax2.imshow(RGB, extent=domain, origin="lower")
            ax2.axis('scaled')
            ax2.axis(domain)
            n_tot = np.sum(num)
            vxm = np.sum(vx) / n_tot
            vym = np.sum(vy) / n_tot
            phi = np.sqrt(vxm**2 + vym**2)

            plt.suptitle(
                title_pat % (self.eta, self.eps, 1, self.Lx, self.seed, t,
                             phi),
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
            if v_normed:
                add_colorbar(cb_ax2, 0, 1, 0, 360)
            else:
                add_colorbar(cb_ax2, 0, 4, 0, 360)
            if savefig:
                plt.savefig(self.fig_dir + os.path.sep + "%04d.png" % i)
            else:
                plt.show()
            plt.close()

    def mk_mv(self, rate=15, start_num=0, vframes=None):
        import subprocess
        strcmd = r"ffmpeg -f image2 -r %d -start_number %d -i %s " % (
            rate, start_num, self.fig_dir + "/%04d.png")
        if vframes is not None:
            strcmd += "-vframes %d %s " % vframes
        strcmd += "-preset veryslow -crf 34 %s" % (self.mp4_file)
        subprocess.call(strcmd, shell=True)

    def movie_exists(self):
        return os.path.exists(self.mp4_file)


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
    V /= m_max
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


def make_animation(eps):
    dest_dir = r"E:\data\random_torque\phase diagram\pd_L=1024"
    if not os.path.exists(dest_dir):
        dest_dir = dest_dir.replace("E", "D")
    os.chdir(dest_dir)
    files = glob.glob(r"cg_*_%g_1_1024_1024_111111.nc" % eps)
    for file in files:
        f = CGSnap(file)
        if f.movie_exists():
            print("%s already exists" % f.mp4_file)
        else:
            f.show(savefig=True, v_normed=False)
            f.mk_mv()


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


def plot_corr_long_tra_varied_eta(eps, L=1024):
    eta_arr = [0.2, 0.21, 0.22, 0.225, 0.23, 0.24, 0.25, 0.26]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    for eta in eta_arr:
        with open("cr_lon_tra_%g_%g_%d.dat" % (eta, eps, L), "r") as f:
            lines = f.readlines()
            n = len(lines)
            r, crho_lon, crho_tra, cv_lon, cv_tra = np.zeros((5, n))
            for i, line in enumerate(lines):
                s = line.replace("\n", "").split("\t")
                r[i] = float(s[0])
                crho_lon[i] = float(s[1])
                crho_tra[i] = float(s[2])
                cv_lon[i] = float(s[3])
                cv_tra[i] = float(s[4])
        axes[0][0].loglog(r, crho_lon, label=r"$%g$" % eta)
        axes[0][1].loglog(r, crho_tra)
        axes[1][0].loglog(r, cv_lon)
        axes[1][1].loglog(r, cv_tra)
    axes[0][0].set_xlabel(r"$r$", fontsize="large")
    axes[0][0].set_ylabel(r"$C_\rho$", fontsize="large")
    axes[0][1].set_xlabel(r"$r$", fontsize="large")
    axes[0][1].set_ylabel(r"$C_\rho$", fontsize="large")
    axes[1][0].set_xlabel(r"$r$", fontsize="large")
    axes[1][0].set_ylabel(r"$C_v$", fontsize="large")
    axes[1][1].set_xlabel(r"$r$", fontsize="large")
    axes[1][1].set_ylabel(r"$C_v$", fontsize="large")
    axes[0][0].set_title("longitudinal")
    axes[0][1].set_title("transversal")
    axes[1][0].set_title("longitudinal")
    axes[1][1].set_title("transversal")
    axes[0][0].legend(title=r"$\eta=$", fontsize="small", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(r"$\epsilon=%g,\ L=%d$" % (eps, L), fontsize="xx-large")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir(r"D:\data\random_torque\phase diagram\pd_L=1024")
    plot_corr_long_tra_varied_eta(0.02)
    # f = Corr2d(r"cr_0.26_0.02_1_1024_1024_111111.nc")
    # f.show_corr_lon_tra(save_data=True)
    # frames = f.gene_frame()
    # for frame in frames:
    #     t, vm, c_rho, c_v = frame
    #     # t, num, vx, vy = frame
    #     print(t, c_v[0, 250])
    #     # plt.imshow(c_v, origin="lower")
    #     # plt.show()
    #     # plt.close()
    # f2 = CGSnap(r"cg_0.25_0.025_1_1024_1024_111111.nc")
    # f2.show(savefig=True, v_normed=False)
    # f2.mk_mv()
    # frames = f2.gene_frame()
    # for frame in frames:
    #     t, num, vx, vy = frame
    #     corr_rho, corr_v = cal_corr2d(num, vx, vy, 1)
    #     print(t, corr_v[0, 250])
    # make_animation(0.02)
