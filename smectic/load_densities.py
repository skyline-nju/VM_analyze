"""
    Read .nc file that records a serials of frames. Each frame contains the
    information of density field and velocity field.
"""

from netCDF4 import Dataset
import glob
import numpy as np
import matplotlib.pyplot as plt
import os


class DensityField_2:
    def __init__(self, file0):
        self.rootgrp = []
        if "host" in file0:
            if "host0" in file0:
                str_arr = file0.split("host0")
            files = glob.glob("%shost*%s" % (str_arr[0], str_arr[1]))
            for file in files:
                self.rootgrp.append(Dataset(file, "r", format="NETCDF4"))
            if len(files) > 1:
                self.host_size = np.array(
                    self.rootgrp[0].variables["host_size"][:], int)
            else:
                self.host_size = np.array([1, 1])
        else:
            self.rootgrp.append(Dataset(file0, "r", format="NETCDF4"))
            self.host_size = np.array([1, 1])
        self.cells_per_host = np.array([
            self.rootgrp[0].dimensions["NY"].size,
            self.rootgrp[0].dimensions["NX"].size
        ], int)
        # cell number for global domain
        self.gl_nc = self.host_size * self.cells_per_host

        keys = ["eta", "Lx", "Ly", "rho_0", "seed"]
        self.para = {key: getattr(self.rootgrp[0], key) for key in keys}
        self.para["dx"] = self.para["Lx"] / self.gl_nc[1]
        self.para["dy"] = self.para["Ly"] / self.gl_nc[0]
        self.para["dA"] = self.para["dx"] * self.para["dy"]
        self.para["N"] = self.para["rho_0"] * self.para["Lx"] * self.para["Ly"]

    def gene_frames(self, first=0, last=None, block_size=1):
        gl_num_count = np.zeros((self.gl_nc[0], self.gl_nc[1]), np.uint16)
        if last is None:
            nframe = self.rootgrp[0].dimensions["frame"].size - first
        else:
            nframe = last - first
        for idx_t in range(first, first + nframe):
            n = len(self.rootgrp)
            if n > 1:
                for i in range(n):
                    host_rank = self.rootgrp[i].variables["host_rank"][:]
                    y_beg = host_rank[0] * self.cells_per_host[0]
                    y_end = y_beg + self.cells_per_host[0]
                    x_beg = host_rank[1] * self.cells_per_host[1]
                    x_end = x_beg + self.cells_per_host[1]
                    gl_num_count[y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables["density_field"][idx_t, :, :]
            else:
                gl_num_count = self.rootgrp[0].variables["density_field"][
                    idx_t, :, :]
            t = self.rootgrp[0].variables["time"][idx_t]
            yield t, gl_num_count

    def plot_one_frame(self,
                       t,
                       rho,
                       title=None,
                       rho_max=None,
                       save_folder=None,
                       rect=False):
        if not rect:
            fig, ax = plt.subplots(
                ncols=1, nrows=1, figsize=(6, 6), constrained_layout=True)
            cb_ori = "horizontal"
        else:
            if self.para["Lx"] / self.para["Ly"] >= 3:
                figsize = (18, 3)
                cb_ori = "vertical"
            else:
                figsize = (12, 5)
                cb_ori = "horizontal"
            fig, ax = plt.subplots(
                ncols=1, nrows=1, figsize=figsize, constrained_layout=True)
        extent = [0, self.para["Lx"], 0, self.para["Ly"]]
        im = ax.imshow(
            rho, origin="lower", extent=extent, vmax=rho_max, aspect="auto")
        cb = plt.colorbar(
            im, ax=ax, orientation=cb_ori, extend="max", aspect=50, pad=0.005)
        cb.set_label(r"$\rho$", fontsize="x-large")

        n_par = self.para["Lx"] * self.para["Ly"] * np.mean(rho)
        print("number of particles: ", n_par)
        if title is None:
            pat = r"$\eta=%g, \rho_0=%g, L_x=%g, L_y=%g, t=%d$"
            title = pat % (self.para["eta"], self.para["rho_0"],
                           self.para["Lx"], self.para["Ly"], t)
        plt.suptitle(title, fontsize="xx-large")
        # plt.tight_layout()
        if save_folder is None:
            plt.show()
        else:
            plt.savefig(save_folder + os.path.sep + "%06d" % t)
        plt.close()

    def plot_frames(self,
                    first=0,
                    last=None,
                    block_size=1,
                    title=None,
                    rho_max=None,
                    save_folder=None,
                    rect=False):
        frames = self.gene_frames(first, last, block_size)
        for i, frame in enumerate(frames):
            t, num = frame
            print("t =", t)
            rho = num / self.para["dA"]
            print("i =", i)
            self.plot_one_frame(t, rho, title, rho_max, save_folder, rect)


if __name__ == "__main__":
    os.chdir(r"data")
    # Lx = 24000
    # for i in range(16):
    #     fname = "field_%d_360_0.35_1.0_1_host0_%d.nc" % (Lx, i)
    #     snap = DensityField_2(fname)
    #     snap.plot_frames(rho_max=8, rect=True, save_folder="Lx%d" % Lx)
    fname = "field_96000_360_0.35_1.0_1_host0_11.nc"
    snap = DensityField_2(fname)
    snap.plot_frames(rho_max=8, rect=True)
