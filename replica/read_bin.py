import numpy as np
import struct
import os


def read_bin(fname, beg=0, end=None, which="both", lbox=4):
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        L = int(os.path.basename(fname).split("_")[2])
        ncols = L // lbox
        nrows = L // lbox
        box_area = lbox * lbox
        n = ncols * nrows
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
                n_mean, vx_mean, vy_mean = np.array(data).reshape(3,
                                                                  n) / box_area
                yield n_mean.reshape(nrows, ncols), vx_mean.reshape(
                    nrows, ncols), vy_mean.reshape(nrows, ncols)
        elif which == "rho":
            while f.tell() < file_end:
                buf = f.read(n * 4)
                f.seek(n * 8, 1)
                data = struct.unpack("%df" % (n), buf)
                n_mean = np.array(data).reshape(nrows, ncols) / box_area
            yield n_mean
        elif which == "v":
            while f.tell() < file_end:
                f.seek(n * 4, 1)
                buf = f.read(n * 8)
                data = struct.unpack("%df" % (n * 2), buf)
                vx_mean, vy_mean = np.array(data).reshape(2, n) / box_area
                yield vx_mean.reshape(nrows,
                                      ncols), vy_mean.reshape(nrows, ncols)


def get_time_averaged_image(fname,
                            beg=0,
                            end=None,
                            which="both",
                            N=128,
                            lbox=4):
    print("n=", N)
    frames = read_bin(fname, beg, end, which, lbox)
    count = 0
    if which == "both":
        rho_mean, vx_mean, vy_mean = np.zeros((3, N, N))
        for rho, vx, vy in frames:
            rho_mean += rho
            vx_mean += vx
            vy_mean += vy
            count += 1
        rho_mean /= count
        vx_mean /= count
        vy_mean /= count
        return rho_mean, vx_mean, vy_mean
    elif which == "rho":
        rho_mean = np.zeros((N, N))
        for frame in frames:
            rho_mean += frame
            count += 1
        rho_mean /= count
        return rho_mean
    elif which == "v":
        vx_mean, vy_mean = np.zeros((2, N, N))
        for vx, vy in frames:
            vx_mean += vx
            vy_mean += vy
            count += 1
        vx_mean /= count
        vy_mean /= count
        return vx_mean, vy_mean
