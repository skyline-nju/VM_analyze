""" Method to read raw or coarse-grained snapshots.

    Raw snapshot record the instant location (x, y) and orientation (theta)
    of each particle with float number (float32).

    Coarse-grained snapshots record the count of number and mean velocity
    over cells with linear size (lx, ly). Types of num, vx, vy are int32,
    float32, float32 for "iff" and unsigned char, signed char, singed char
    for "Bbb". Additional imformation such as time step, sum of vx and vy would
    be also saved in the file.

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
        str_list = file.split("_")
        if str_list[0] == "so":
            self.N = int(str_list[5])
        else:
            self.N = self.file_size // 12
        self.Lx = int(str_list[3])
        self.Ly = int(str_list[4])
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
        self.snap_format = str_list[0].replace("c", "")
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
        self.frame_size = self.snap_size + 20
        self.open_file(file)

    def one_frame(self):
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
            num = np.array(data[:self.N], int).reshape(self.nrows, self.ncols)
            vx = np.array(data[self.N:2 * self.N], float).reshape(
                self.nrows, self.ncols)
            vy = np.array(data[2 * self.N:3 * self.N], float).reshape(
                self.nrows, self.ncols)
            if self.snap_format == "Bbb":
                vx /= 128
                vy /= 128
            frame = [t, vxm, vym, num, vx, vy]
        return frame


if __name__ == "__main__":
    # os.chdir(r"D:\code\corr2d\data")
    os.chdir(r"D:\code\VM_MPI\VM_MPI\coarse")
    # os.chdir(r"E:\data\random_torque\ordering")
    file = r"cHff_0.18_0.02_512_512_128_128_262144_1.bin"
    snap = CoarseGrainSnap(file)
    frames = snap.gene_frames(beg_idx=0)
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        print("t=%d\tdN=%d\tdvx=%f\tdvy=%f" %
              (t, np.sum(num) - 512 * 512,
               np.sum(vx * num) / np.sum(num) - vxm,
               np.sum(vy * num) / np.sum(num) - vym))

    # file = r"cB_0.18_0.02_512_512_256_256_262144_2.bin"
    # snap = CoarseGrainSnap(file)
    # frames = snap.gene_frames(beg_idx=0)
    # for frame in frames:
    #     t, vxm, vym, num = frame
    #     print(t, np.sum(num), 512 * 512)
