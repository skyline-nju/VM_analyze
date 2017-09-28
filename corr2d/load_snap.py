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
from scipy.ndimage import gaussian_filter

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

    def show(self,
             i_beg=0,
             i_end=None,
             di=1,
             lx=1,
             ly=1,
             transpos=True,
             sigma=[5, 1],
             output=True,
             show=True):
        import half_rho
        import half_peak
        if output:
            f = open(self.file.replace(".bin", "_%d.dat" % (sigma[0])), "w")
        for i, frame in enumerate(self.gene_frames(i_beg, i_end, di)):
            t, vxm, vym, num = frame
            rho = num.astype(np.float32)
            yh = np.linspace(0.5, self.nrows - 0.5, self.nrows)
            # xh1, rho_h1 = half_rho.find_interface(rho, sigma=sigma)
            xh2, rho_h2 = half_peak.find_interface(rho, sigma=sigma)
            # xh1 = half_rho.untangle(xh1, self.ncols)
            xh2 = half_rho.untangle(xh2, self.ncols)
            # w1 = np.var(xh1)
            w2 = np.var(xh2)
            if show:
                if ly > 1:
                    rho = np.array([
                        np.mean(num[i * ly:(i + 1) * ly], axis=0)
                        for i in range(self.nrows // ly)
                    ])
                if lx > 1:
                    rho = np.array([
                        np.mean(rho[:, i * lx:(i + 1) * lx], axis=1)
                        for i in range(self.ncols // lx)
                    ])
                rho = gaussian_filter(rho, sigma=[5, 1])
                if transpos:
                    rho = rho.T
                    box = [0, self.nrows, 0, self.ncols]
                    plt.figure(figsize=(14, 3))
                    plt.plot(yh, xh2, "r")
                    plt.xlabel(r"$y$")
                    plt.ylabel(r"$x$")
                else:
                    box = [0, self.ncols, 0, self.nrows]
                    plt.figure(figsize=(4, 12))
                    plt.plot(xh2, yh, "r")
                    plt.xlabel(r"$x$")
                    plt.ylabel(r"$y$")
                rho[rho > 4] = 4
                plt.imshow(
                    rho,
                    origin="lower",
                    interpolation="none",
                    extent=box,
                    aspect="auto")
                plt.title(r"$t=%d, \phi=%g, w^2=%g$" %
                          (t, np.sqrt(vxm**2 + vym**2), w2))
                plt.tight_layout()
                plt.show()
                plt.close()
            print(t, np.sqrt(vxm**2 + vym**2), w2)
            if output:
                f.write("%d\t%f\t%f\n" % (t, np.sqrt(vxm**2 + vym**2), w2))
        if output:
            f.close()


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
