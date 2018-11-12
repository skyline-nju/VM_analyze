"""
    Read .nc file that records a serials of frames. Each frame contains the
    information of density field and velocity field.
"""

from netCDF4 import Dataset
import glob
import numpy as np
import os
import cpp2py.num_flct as n_flct

def get_grp(file0):
    grp = []
    if "host" in file0:
        pat = file0[:-5]
        files = glob.glob("%s*.nc" % pat)
    else:
        files = [file0]
    for file in files:
        grp.append(Dataset(file, "r", format="NETCDF4"))
    return grp


def get_host_info(file0):
    grp = Dataset(file0, "r", format="NCTCDF4")
    if "host" in file0:
        is_mult_files = True
    else:
        is_mult_files = False
    if grp.dimensions["spatial"].size == 2:
        if is_mult_files:
            host_size = np.array(grp.variables["host_size"][:], int)
        else:
            host_size = [1, 1]
        cells_per_host = np.array([
            grp.dimensions["NY"].size,
            grp.dimensions["NX"].size], int)
    else:
        if is_mult_files:
            host_size = np.array(grp.variables["host_size"][:], int)
        else:
            host_size = [1, 1, 1]
        if "NZ" in grp.dimensions:
            cells_per_host = np.array([
                grp.dimensions["NZ"].size,
                grp.dimensions["NY"].size,
                grp.dimensions["NX"].size], int)
        else:
            cells_per_host = np.array([
                grp.dimensions["global_field_z"].size,
                grp.dimensions["global_field_y"].size,
                grp.dimensions["global_field_x"].size], int)
    grp.close()
    return host_size, cells_per_host


def read_densities_2(file0, first_frame=0):
    rootgrp = get_grp(file0)
    host_size, cells_per_host = get_host_info(file0)
    gl_cells = host_size * cells_per_host
    gl_density = np.zeros((gl_cells[0], gl_cells[1]), np.uint16)
    nframe = rootgrp[0].dimensions["frame"].size
    for idx_t in range(first_frame, nframe):
        n = len(rootgrp)
        if n > 1:
            for i in range(n):
                host_rank = rootgrp[i].variables["host_rank"][:]
                y_beg = host_rank[0] * cells_per_host[0]
                y_end = y_beg + cells_per_host[0]
                x_beg = host_rank[1] * cells_per_host[1]
                x_end = x_beg + cells_per_host[1]
                gl_density[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["density_field"][idx_t, :, :]
        else:
            gl_density = rootgrp[0].variables["density_field"][idx_t, :, :]
        t = rootgrp[0].variables["time"][idx_t]
        yield t, gl_density


def read_velocities_2(file0, first_frame=0):
    rootgrp = get_grp(file0)
    host_size, cells_per_host = get_host_info(file0)
    gl_cells = host_size * cells_per_host
    gl_vx = np.zeros((gl_cells[0], gl_cells[1]))
    gl_vy = np.zeros_like(gl_vx)
    nframe = rootgrp[0].dimensions["frame"].size
    for idx_t in range(first_frame, nframe):
        n = len(rootgrp)
        if n > 1:
            for i in range(n):
                host_rank = rootgrp[i].variables["host_rank"][:]
                y_beg = host_rank[0] * cells_per_host[0]
                y_end = y_beg + cells_per_host[0]
                x_beg = host_rank[1] * cells_per_host[1]
                x_end = x_beg + cells_per_host[1]
                gl_vy[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["velocity_field"][idx_t, 0, :, :]
                gl_vx[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["velocity_field"][idx_t, 1, :, :]
        else:
            gl_vx = rootgrp[0].variables["velocity_field"][idx_t, 0, :, :]
            gl_vy = rootgrp[0].variables["velocity_field"][idx_t, 1, :, :]
        t = rootgrp[0].variables["time"][idx_t]
        yield t, gl_vx, gl_vy


def read_2(file0, first_frame=0):
    rootgrp = get_grp(file0)
    host_size, cells_per_host = get_host_info(file0)
    gl_cells = host_size * cells_per_host
    gl_density = np.zeros((gl_cells[0], gl_cells[1]), np.uint16)
    gl_vx = np.zeros((gl_cells[0], gl_cells[1]))
    gl_vy = np.zeros_like(gl_vx)
    nframe = rootgrp[0].dimensions["frame"].size
    for idx_t in range(first_frame, nframe):
        n = len(rootgrp)
        if n > 1:
            for i in range(n):
                host_rank = rootgrp[i].variables["host_rank"][:]
                y_beg = host_rank[0] * cells_per_host[0]
                y_end = y_beg + cells_per_host[0]
                x_beg = host_rank[1] * cells_per_host[1]
                x_end = x_beg + cells_per_host[1]
                gl_density[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["density_field"][idx_t, :, :]
                gl_vx[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["velocity_field"][idx_t, 0, :, :]
                gl_vy[y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["velocity_field"][idx_t, 1, :, :]
        else:
            gl_density = rootgrp[0].variables["density_field"][idx_t, :, :]
            gl_vx = rootgrp[0].variables["velocity_field"][idx_t, 0, :, :]
            gl_vy = rootgrp[0].variables["velocity_field"][idx_t, 1, :, :]
        t = rootgrp[0].variables["time"][idx_t]
        yield t, gl_density, gl_vx, gl_vy


def read_densities_3(file0, first_frame):
    rootgrp = get_grp(file0)
    host_size, cells_per_host = get_host_info(file0)
    gl_cells = host_size * cells_per_host
    gl_density = np.zeros((gl_cells[0], gl_cells[1], gl_cells[2]), np.uint16)
    nframe = rootgrp[0].dimensions["frame"].size
    for idx_t in range(first_frame, nframe):
        n = len(rootgrp)
        if n > 1:
            for i in range(n):
                host_rank = rootgrp[i].variables["host_rank"][:]
                z_beg = host_rank[0] * cells_per_host[0]
                z_end = z_beg + cells_per_host[0]
                y_beg = host_rank[1] * cells_per_host[1]
                y_end = y_beg + cells_per_host[1]
                x_beg = host_rank[2] * cells_per_host[2]
                x_end = x_beg + cells_per_host[2]
                gl_density[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["density_field"][idx_t, :, :, :]
        else:
            gl_density = rootgrp[0].variables["density_field"][idx_t, :, :, :]
        t = rootgrp[0].variables["time"][idx_t]
        yield t, gl_density


def read_3(file0, first_frame):
    rootgrp = get_grp(file0)
    host_size, cells_per_host = get_host_info(file0)
    gl_cells = host_size * cells_per_host
    gl_density = np.zeros((gl_cells[0], gl_cells[1], gl_cells[2]), np.uint16)
    gl_vx = np.zeros((gl_cells[0], gl_cells[1], gl_cells[2]))
    gl_vy = np.zeros_like(gl_vx)
    gl_vz = np.zeros_like(gl_vx)
    nframe = rootgrp[0].dimensions["frame"].size
    key_v = "spatial_field"
    for idx_t in range(first_frame, nframe):
        n = len(rootgrp)
        if n > 1:
            for i in range(n):
                host_rank = rootgrp[i].variables["host_rank"][:]
                z_beg = host_rank[0] * cells_per_host[0]
                z_end = z_beg + cells_per_host[0]
                y_beg = host_rank[1] * cells_per_host[1]
                y_end = y_beg + cells_per_host[1]
                x_beg = host_rank[2] * cells_per_host[2]
                x_end = x_beg + cells_per_host[2]
                gl_density[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables["density_field"][idx_t, :, :, :]
                gl_vx[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables[key_v][idx_t, 0, :, :, :]
                gl_vy[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables[key_v][idx_t, 1, :, :, :]
                gl_vz[z_beg: z_end, y_beg: y_end, x_beg: x_end] = \
                    rootgrp[i].variables[key_v][idx_t, 2, :, :, :]
        else:
            gl_density = rootgrp[0].variables["density_field"][idx_t, :, :, :]
            gl_vx = rootgrp[0].variables[key_v][idx_t, 0, :, :, :]
            gl_vy = rootgrp[0].variables[key_v][idx_t, 1, :, :, :]
            gl_vz = rootgrp[0].variables[key_v][idx_t, 2, :, :, :]
        t = rootgrp[0].variables["time"][idx_t]
        yield t, gl_density, gl_vx, gl_vy, gl_vz


def coarse_grain_2(X, block_size=2):
    ny, nx = X.shape
    ny_new, nx_new = ny // block_size, nx // block_size
    if X.dtype == np.uint16 or X.dtype == np.int32:
        Y = np.zeros((ny_new, nx_new), np.int32)
    else:
        Y = np.zeros((ny_new, nx_new))
    for j in range(ny_new):
        y_beg = j * block_size
        y_end = y_beg + block_size
        for i in range(nx_new):
            x_beg = i * block_size
            x_end = x_beg + block_size
            Y[j, i] = np.sum(X[y_beg: y_end, x_beg: x_end])
    return Y


class Snapshot_2:
    def __init__(self, file0):
        self.rootgrp = []
        if "host" in file0:
            pat = file0[:-5]
            files = glob.glob("%s*.nc" % pat)
            for file in files:
                self.rootgrp.append(Dataset(file, "r", format="NETCDF4"))
            self.host_size = np.array(self.rootgrp[0].variables["host_size"][:], int)
        else:
            self.rootgrp.append(Dataset(file0, "r", format="NETCDF4"))
            self.host_size = [1, 1]
        self.cells_per_host = np.array([
            self.rootgrp[0].dimensions["NY"].size,
            self.rootgrp[0].dimensions["NX"].size], int)
        str_list = os.path.basename(file0).split("_")
        self.L = int(str_list[1])
        self.eta = float(str_list[2])
        self.eps = float(str_list[3])

    def gene_frames(self, first=0, last=None, block_size=1):
        gl_cells = self.host_size * self.cells_per_host
        gl_num_count = np.zeros((gl_cells[0], gl_cells[1]), np.uint16)
        gl_vx = np.zeros((gl_cells[0], gl_cells[1]))
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
                        self.rootgrp[i].variables["velocity_field"][idx_t, 0, :, :]
                    gl_vy[y_beg: y_end, x_beg: x_end] = \
                        self.rootgrp[i].variables["velocity_field"][idx_t, 1, :, :]
            else:
                gl_num_count = self.rootgrp[0].variables["density_field"][idx_t, :, :]
                gl_vx = self.rootgrp[0].variables["velocity_field"][idx_t, 0, :, :]
                gl_vy = self.rootgrp[0].variables["velocity_field"][idx_t, 1, :, :]
            t = self.rootgrp[0].variables["time"][idx_t]
            if block_size == 1:
                yield t, gl_num_count, gl_vx, gl_vy
            else:
                ny_new = gl_cells[0] // block_size
                nx_new = gl_cells[1] // block_size
                gl_num_count_new = np.zeros((ny_new, nx_new), np.int32)
                gl_vx_new, gl_vy_new = np.zeros((2, ny_new, nx_new))
                n_flct.renormalize_2d_uint16(gl_num_count, gl_num_count_new)
                n_flct.renormalize_2d_doub(gl_vx, gl_vx_new)
                n_flct.renormalize_2d_doub(gl_vy, gl_vy_new)
                yield t, gl_num_count_new, gl_vx_new, gl_vy_new
