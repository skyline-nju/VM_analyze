""" Method to read raw or coarse-grained snapshots.

    `RawSnap` handle the binary file recording the time serials of coordinates
    `x, y, theta` in `float32` format.

    `CoarseGrainSnap` handle the binary file recording the time serials of
    density and velocity fields.

    FIND A BUG:
    code:
    --------
        f = open(file, "rb")
        buff = f.read(20)
        a = struct.unpack('idd', buff)

    output:
    --------
        struct.error: unpack requires a bytes object of length 24
"""
# import glob
import os
import sys
import struct
import numpy as np
import platform
import matplotlib

if platform.system() is not "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


class Snap:
    """ Base class for snapshot. """

    def __init__(self, file):
        """ Need rewrite for subclass. """
        self.open_file(file)

    def __del__(self):
        self.f.close()

    def open_file(self, file):
        self.f = open(file, "rb")
        self.f.seek(0, 2)
        self.file_size = self.f.tell()
        self.f.seek(0)
        print("open ", file)

    def one_frame(self):
        """ Need rewrite for subclass. """
        pass

    def read_frame(self, idx=0):
        offset = idx * self.frame_size
        if self.file_size - offset >= self.frame_size:
            self.f.seek(offset)
            return self.one_frame()
        else:
            print("Error, index of frame should be less than %d" %
                  (self.file_size // self.frame_size))
            sys.exit()

    def gene_frames(self, beg_idx=0, end_idx=None, interval=1, frames=None):
        """ Generator of frames.

        Parameters:
        --------
        beg_idx: int, optional
            The first frame to read.
        end_idx: int, optional
            The last frame.
        interval: int, optional
            The interval between two frames.
        frames: array_like, optional
            The list of index of frames to read.

        Yields:
        --------
        frame: array_like
            One frame containing the instant information.
        """
        if frames is None:
            self.f.seek(beg_idx * self.frame_size)
            if end_idx is None:
                max_size = self.file_size
            else:
                max_size = end_idx * self.frame_size
            count = 0
            while max_size - self.f.tell() >= self.frame_size:
                if count % interval == 0:
                    yield self.one_frame()
                else:
                    self.f.seek(self.frame_size, 1)
                count += 1
        else:
            for i in frames:
                self.f.seek(i * self.frame_size)
                yield self.one_frame()

    def get_tot_frames_num(self):
        return self.file_size // self.frame_size


class RawSnap(Snap):
    def __init__(self, file):
        self.open_file(file)
        str_list = os.path.basename(file).split("_")
        if str_list[0] == "so":
            self.N = int(str_list[5])
        else:
            self.N = self.file_size // 12
        try:
            self.Lx = int(str_list[3])
            self.Ly = int(str_list[4])
        except ValueError:
            self.Lx = self.Ly = int(str_list[1])
        self.frame_size = self.N * 3 * 4
        self.fmt = "%df" % (3 * self.N)

    def one_frame(self):
        buff = self.f.read(self.frame_size)
        data = struct.unpack(self.fmt, buff)
        frame = np.array(data, float).reshape(self.N, 3).T
        return frame

    def show(self, beg_idx=0, end_idx=None, interval=1, markersize=1):
        for i, frame in enumerate(
                self.gene_frames(beg_idx, end_idx, interval)):
            x, y, theta = frame
            plt.plot(x, y, "o", ms=markersize)
            plt.title("frame %d" % (interval * i))
            plt.show()
            plt.close()


class CoarseGrainSnap(Snap):
    def __init__(self, file):
        str_list = file.split("_")
        self.snap_format = str_list[0][1:]
        if (str_list[0][0] == "c"):
            self.is_continous = False
        else:
            self.is_continous = True
        self.ncols = int(str_list[5])
        self.nrows = int(str_list[6])
        self.N = self.ncols * self.nrows
        self.file = file
        print(self.ncols, self.nrows)
        if self.snap_format == "Bbb":
            self.fmt = "%dB%db" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3
        elif self.snap_format == "iff":
            self.fmt = "%di%df" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3 * 4
        elif self.snap_format == "Hff":
            self.fmt = "%dH%df" % (self.N, 2 * self.N)
            self.snap_size = self.N * 10
        elif self.snap_format == "B":
            self.fmt = "%dB" % (self.N)
            self.snap_size = self.N
        elif self.snap_format == "fff":
            self.fmt = "%df" % (self.N * 3)
            self.snap_size = self.N * 3 * 4
        if self.snap_format == "fff":
            self.frame_size = self.snap_size
        else:
            if self.is_continous:
                self.frame_size = self.snap_size + 24
            else:
                self.frame_size = self.snap_size + 20
        self.open_file(file)
        print(file)

    def one_frame(self):
        if self.snap_format == "fff":
            buff = self.f.read(self.snap_size)
            data = struct.unpack(self.fmt, buff)
            num = np.array(data[:self.N], float).reshape(
                self.nrows, self.ncols)
            vx = np.array(data[self.N:self.N * 2], float).reshape(
                self.nrows, self.ncols)
            vy = np.array(data[self.N * 2:self.N * 3], float).reshape(
                self.nrows, self.ncols)
            frame = [num, vx, vy]
        else:
            if self.is_continous:
                buff = self.f.read(24)
                t, vxm, vym = struct.unpack("ddd", buff)
            else:
                buff = self.f.read(4)
                t, = struct.unpack("i", buff)
                buff = self.f.read(16)
                vxm, vym = struct.unpack("dd", buff)
            buff = self.f.read(self.snap_size)
            data = struct.unpack(self.fmt, buff)
            if self.snap_format == "B":
                num = np.array(data, int).reshape(self.nrows, self.ncols)
                frame = [t, vxm, vym, num]
            else:
                num = np.array(data[:self.N], int).reshape(
                    self.nrows, self.ncols)
                vx = np.array(data[self.N:2 * self.N], float).reshape(
                    self.nrows, self.ncols)
                vy = np.array(data[2 * self.N:3 * self.N], float).reshape(
                    self.nrows, self.ncols)
                if self.snap_format == "Bbb":
                    vx /= 128
                    vy /= 128
                frame = [t, vxm, vym, num, vx, vy]
        return frame


def coarse_grain(x, y, theta, Lx, lx, Ly=None, ly=None):
    if Ly is None:
        Ly = Lx
    if ly is None:
        ly = lx
    ncols, nrows = int(Lx / lx), int(Ly / ly)
    num = np.zeros((nrows, ncols), int)
    vx = np.zeros((nrows, ncols))
    vy = np.zeros_like(vx)
    for i in range(x.size):
        col = int(x[i]/lx)
        if col >= ncols:
            col = 0
        row = int(y[i]/ly)
        if row >= nrows:
            row = 0
        num[row, col] += 1
        vx[row, col] += np.cos(theta[i])
        vy[row, col] += np.sin(theta[i])
    return num, vx, vy


if __name__ == "__main__":
    file_dir = "D:" + r"\data\random_torque\snapshot" + os.path.sep
    basename = "s_1024_0.180_0.040_1204400_01300000.bin"
    snap = RawSnap(file_dir + basename)
    L = 1024
    l_box = 2
    x, y, theta = snap.one_frame()
    num, vx, vy = coarse_grain(x, y, theta, L, l_box)
    rho = num / 4
    print(rho.max())

    # plt.scatter(x, y, s=1, c=theta, marker=".", cmap="hsv")
    # plt.show()
    # plt.plot(x[5000:10000], y[5000:10000], ".")
    # plt.show()
    # num, vx, vy = coarse_grain(x, y, theta, L, l_box)
    # rho = num / l_box ** 2
    # print(np.std(rho), np.mean(rho))
    # rho_level = np.linspace(0, 10, 11)
    # plt.contourf(rho, levels=rho_level, cmap="jet")
    # plt.colorbar()
    # plt.show()
