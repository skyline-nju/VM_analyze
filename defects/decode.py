import numpy as np
import struct
import os


def get_nframe(fname, lbox=4):
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        basename = os.path.basename(fname)
        if "RT_ave" in basename or "RT_f" in basename:
            L = int(basename.split("_")[2])
            n = L * L // (lbox * lbox)
            framesize = n * 12
            n_frame = filesize // framesize
    return n_frame


def read_time_ave_bin(fname, beg=0, end=None, which="both", lbox=4):
    """ Read time-averaged data saved as binary format."""
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        L = int(os.path.basename(fname).split("_")[2])
        nx, ny = L // lbox, L // lbox
        n = L * L // (lbox * lbox)
        framesize = n * 12
        f.seek(beg * framesize)
        if end is None:
            file_end = filesize
        else:
            file_end = end * framesize
        if which == "both":
            while f.tell() < file_end:
                buf = f.read(framesize)
                data = struct.unpack("%df" % (n * 3), buf)
                n_mean, vx_mean, vy_mean = np.array(data).reshape(
                    3, n) / (lbox * lbox)
                yield n_mean.reshape(ny, nx), vx_mean.reshape(
                    ny, nx), vy_mean.reshape(ny, nx)
        elif which == "rho":
            while f.tell() < file_end:
                buf = f.read(n * 4)
                f.seek(n * 8, 1)
                data = struct.unpack("%df" % (n), buf)
                n_mean = np.array(data).reshape(ny, nx) / (lbox * lbox)
            yield n_mean
        elif which == "v":
            while f.tell() < file_end:
                f.seek(n * 4, 1)
                buf = f.read(n * 8)
                data = struct.unpack("%df" % (n * 2), buf)
                vx_mean, vy_mean = np.array(data).reshape(2, n) / (lbox * lbox)
                yield vx_mean.reshape(ny, nx), vy_mean.reshape(ny, nx)


def read_field(fname, beg=0, end=None, sep=1, lbox=4):
    """ Read coarse-grained density and momentum fields """
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        L = int(os.path.basename(fname).split("_")[2])
        nx, ny = L // lbox, L // lbox
        n = L * L // (lbox * lbox)
        framesize = n * 12
        f.seek(beg * framesize)
        if end is None:
            file_end = filesize
        else:
            file_end = end * framesize
        idx_frame = 0
        while f.tell() < file_end:
            if idx_frame % sep == 0:
                buf = f.read(framesize)
                data = np.array(struct.unpack("%df" % (n * 3), buf))
                rho_m, vx_m, vy_m = data.reshape(3, ny, nx) / (lbox * lbox)
                yield rho_m, vx_m, vy_m
            else:
                f.seek(framesize, 1)
            idx_frame += 1


def cal_order_para(L, eta, eps, seed, theta0, ncut=4000):
    fin = "order_para/p%d.%g.%g.%d.%d.dat" % (L, eta * 1000, eps * 1000, seed,
                                              theta0)
    with open(fin, "r") as f:
        lines = f.readlines()
        phi_arr = np.array([float(i.split("\t")[0]) for i in lines[ncut:]])
        phi_mean = np.mean(phi_arr)
    return phi_mean
