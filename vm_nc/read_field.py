"""
    Read .nc file that records a serials of frames. Each frame contains the
    information of density field and velocity field.
"""

from netCDF4 import Dataset
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import cpp2py.num_flct as n_flct


class Field_2:
    def __init__(self, file0):
        self.rootgrp = []
        if "host" in file0:
            pat = file0[:-5]
            files = glob.glob("%s*.nc" % pat)
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

        keys = ["eta", "epsilon", "Lx", "Ly", "rho_0", "seed"]
        self.para = {key: getattr(self.rootgrp[0], key) for key in keys}
        self.para["dx"] = self.para["Lx"] / self.gl_nc[1]
        self.para["dy"] = self.para["Ly"] / self.gl_nc[0]
        self.para["dA"] = self.para["dx"] * self.para["dy"]
        self.para["N"] = self.para["rho_0"] * self.para["Lx"] * self.para["Ly"]

    def gene_frames(self, first=0, last=None, block_size=1):
        gl_num_count = np.zeros((self.gl_nc[0], self.gl_nc[1]), np.uint16)
        gl_vx = np.zeros((self.gl_nc[0], self.gl_nc[1]))
        gl_vy = np.zeros_like(gl_vx)
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
                    gl_vx[y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables["velocity_field"][idx_t,
                                                                    0, :, :]
                    gl_vy[y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables["velocity_field"][idx_t,
                                                                    1, :, :]
            else:
                gl_num_count = self.rootgrp[0].variables["density_field"][
                    idx_t, :, :]
                gl_vx = self.rootgrp[0].variables["velocity_field"][idx_t,
                                                                    0, :, :]
                gl_vy = self.rootgrp[0].variables["velocity_field"][idx_t,
                                                                    1, :, :]
            t = self.rootgrp[0].variables["time"][idx_t]
            if block_size == 1:
                yield t, gl_num_count, gl_vx, gl_vy
            else:
                ny_new = self.gl_nc[0] // block_size
                nx_new = self.gl_nc[1] // block_size
                gl_num_count_new = np.zeros((ny_new, nx_new), np.int32)
                gl_vx_new, gl_vy_new = np.zeros((2, ny_new, nx_new))
                n_flct.renormalize_2d_uint16(gl_num_count, gl_num_count_new)
                n_flct.renormalize_2d_doub(gl_vx, gl_vx_new)
                n_flct.renormalize_2d_doub(gl_vy, gl_vy_new)
                yield t, gl_num_count_new, gl_vx_new, gl_vy_new

    def plot_one_frame(self, t, rho, vx, vy, title=None):
        fig, (ax1, ax2) = plt.subplots(
            ncols=2, nrows=1, figsize=(8, 5), constrained_layout=True)
        extent = [0, self.para["Lx"], 0, self.para["Ly"]]
        im1 = ax1.imshow(rho, origin="lower", extent=extent)
        cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal")
        cb1.set_label(r"$\rho$", fontsize="x-large")

        theta = np.arctan2(vy, vx)
        im2 = ax2.imshow(theta, origin="lower", extent=extent, cmap="hsv")
        cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")
        cb2.set_label(r"$\theta$", fontsize="x-large")

        vx_m = np.sum(vx) / self.para["N"]
        vy_m = np.sum(vy) / self.para["N"]
        order_para = np.sqrt(vx_m**2 + vy_m**2)
        if title is None:
            pat = r"$\eta=%g,\epsilon=%g, \rho_0=%g, L=%g, t=%d, \phi=%.4f$"
            title = pat % (self.para["eta"], self.para["epsilon"],
                           self.para["rho_0"], self.para["Lx"], t, order_para)
        plt.suptitle(title, fontsize="xx-large")
        plt.show()
        plt.close()
    
    def plot_frames(self, first=0, last=None, block_size=1, title=None):
        frames = self.gene_frames(first, last, block_size)
        for i, frame in enumerate(frames):
            t, num, vx, vy = frame
            rho = num / self.para["dA"]
            self.plot_one_frame(t, rho, vx, vy, title)


class Field_3:
    def __init__(self, file0):
        self.rootgrp = []
        if "host" in file0:
            pat = file0[:-5]
            files = glob.glob("%s*.nc" % pat)
            for file in files:
                self.rootgrp.append(Dataset(file, "r", format="NETCDF4"))
            self.host_size = np.array(
                self.rootgrp[0].variables["host_size"][:], int)
            self.cells_per_host = np.array([
                self.rootgrp[0].dimensions["NZ"].size,
                self.rootgrp[0].dimensions["NY"].size,
                self.rootgrp[0].dimensions["NX"].size
            ], int)
        else:
            self.rootgrp.append(Dataset(file0, "r", format="NETCDF4"))
            self.host_size = [1, 1, 1]
            self.cells_per_host = np.array([
                self.rootgrp[0].dimensions["global_field_z"].size,
                self.rootgrp[0].dimensions["global_field_y"].size,
                self.rootgrp[0].dimensions["global_field_x"].size
            ], int)
        str_list = os.path.basename(file0).replace(".nc", "").split("_")
        self.L = int(str_list[1])
        self.eta = float(str_list[2])
        self.eps = float(str_list[3])
        self.rho0 = float(str_list[4])
        self.seed = int(str_list[5])
        # cell number for global domain
        self.gl_nc = self.host_size * self.cells_per_host

    def gene_frames(self, first=0, last=None, block_size=1):
        gl_num_count = np.zeros((self.gl_nc[0], self.gl_nc[1], self.gl_nc[2]),
                                np.uint16)
        gl_vx = np.zeros((self.gl_nc[0], self.gl_nc[1], self.gl_nc[2]),
                         np.uint16)
        gl_vy = np.zeros_like(gl_vx)
        gl_vz = np.zeros_like(gl_vx)
        key_v = "spatial_field"
        if last is None:
            nframe = self.rootgrp[0].dimensions["frame"].size - first
        else:
            nframe = last - first
        for idx_t in range(first, first + nframe):
            n = len(self.rootgrp)
            if n > 1:
                for i in range(n):
                    host_rank = self.rootgrp[i].variables["host_rank"][:]
                    z_beg = host_rank[0] * self.cells_per_host[0]
                    z_end = z_beg + self.cells_per_host[0]
                    y_beg = host_rank[1] * self.cells_per_host[1]
                    y_end = y_beg + self.cells_per_host[1]
                    x_beg = host_rank[2] * self.cells_per_host[2]
                    x_end = x_beg + self.cells_per_host[2]
                    gl_num_count[z_beg: z_end,
                                 y_beg: y_end,
                                 x_beg: x_end] = \
                        self.rootgrp[i].variables["density_field"][idx_t,
                                                                   :, :, :]
                    gl_vx[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables[key_v][idx_t, 0, :, :, :]
                    gl_vy[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables[key_v][idx_t, 1, :, :, :]
                    gl_vz[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables[key_v][idx_t, 2, :, :, :]
            else:
                gl_num_count = self.rootgrp[0].variables["density_field"][
                    idx_t, :, :, :]
                gl_vx = self.rootgrp[0].variables[key_v][idx_t, 0, :, :, :]
                gl_vy = self.rootgrp[0].variables[key_v][idx_t, 1, :, :, :]
                gl_vz = self.rootgrp[0].variables[key_v][idx_t, 2, :, :, :]
            t = self.rootgrp[0].variables["time"][idx_t]
            if block_size == 1:
                yield t, gl_num_count, gl_vx, gl_vy, gl_vz
            else:
                nz_new = self.gl_nc[0] // block_size
                ny_new = self.gl_nc[1] // block_size
                nx_new = self.gl_nc[2] // block_size
                gl_num_count_new = np.zeros((nz_new, ny_new, nx_new), np.int32)
                gl_vx_new, gl_vy_new, gl_vz_new = np.zeros((3, nz_new, ny_new,
                                                            nx_new))
                n_flct.renormalize_3d_uint16(gl_num_count, gl_num_count_new)
                n_flct.renormalize_3d_doub(gl_vx, gl_vx_new)
                n_flct.renormalize_3d_doub(gl_vy, gl_vy_new)
                n_flct.renormalize_3d_doub(gl_vz, gl_vz_new)
                yield t, gl_num_count_new, gl_vx_new, gl_vy_new, gl_vz_new


if __name__ == "__main__":
    os.chdir(r"E:\data\vm3d\field")
    fname = "field_240_0.20_0.060_1.0_12.nc"
    snap = Field_3(fname)
    frames = snap.gene_frames(0, 10, block_size=1)
    for frame in frames:
        t, num_count, vx, vy, vz = frame
        print(t, np.mean(vx))
