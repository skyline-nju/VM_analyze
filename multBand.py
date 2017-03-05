import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os
import sys


def check_gap(xs, dx_min, Lx):
    """ Check whether dx = xs[i] - xs[i-1] > dx_min.

        Parameters:
        --------
            xs: list
                The list to be checked, note that
                0 < xs[0] < xs[1] < ... < xs[n-2] < xs[n-1] < Lx.
            dx_min: float
                The minimum distance between two nearest xs
            Lx: float
                The superior limit of x in xs

        Returns:
        --------
            xs: list
                Modified list
    """
    if len(xs) <= 2:
        return xs
    else:
        i_head = 0
        is_find_head = False
        while i_head < len(xs) - 1:
            dx = xs[i_head + 1] - xs[i_head]
            if dx >= dx_min:
                is_find_head = True
                break
            else:
                i_head += 1
        if i_head == len(xs) - 1:
            dx = xs[0] + Lx - xs[-1]
            if dx >= dx_min:
                is_find_head = True
        if not is_find_head:
            return []
        else:
            xs = [xs[i + i_head - len(xs)] for i in range(len(xs))]
            x_pre = xs[0]
            i = len(xs) - 1
            while i >= 0 and len(xs) > 1:
                dx = x_pre - xs[i]
                if dx < 0:
                    dx += Lx
                if dx < dx_min and len(xs) > 1:
                    del xs[i]
                else:
                    x_pre = xs[i]
                i -= 1
            return xs


def locatePeak(rho_x, Lx, h=1.8):
    """ Count number of peaks and find x where rho_x=h.

        Parameters:
        --------
            rho_x: np.ndarray
                Density profile as a function of x
            Lx: int
                Length of rho_x
            h: float
                The slope of peak at rho_x=h should be very steep

        Return:
        --------
            xPeak: np.ndarray
                Array of x where rho_x=h
    """
    n = rho_x.size
    x = np.arange(n) + 0.5
    xPeak = []
    for i in range(n):
        if rho_x[i - 1] > h and \
                rho_x[i] <= h and \
                rho_x[(i + 10) % n] < h and \
                rho_x[(i + 20) % n] < h and \
                rho_x[(i + 30) % n] < h and \
                rho_x[(i + 40) % n] < h:
            if i == 0:
                x_left = x[n - 1] - Lx
            else:
                x_left = x[i - 1]
            x_right = x[i]
            xp = x_left - (rho_x[i - 1] - h) / (rho_x[i - 1] - rho_x[i]) * (
                x_left - x_right)
            if xp < 0:
                xp += Lx
            xPeak.append(xp)

    xPeak = check_gap(xPeak, 10, Lx)
    xPeak = check_gap(xPeak, 100, Lx)
    return np.array(xPeak)


class TimeSerialsPeak:
    """ class for time serials of peaks.

        Instance variables:
        --------
            self.xs: np.ndarray
                Time serials of locations of peaks.
            self.num_raw: np.ndarray
                Raw time serials of peak numbers.
            self.num_smoothed: np.ndarray
                Smoothed time serials of peak numbers.
    """

    def __init__(self, file, Lx, beg, h):
        """ Initialize class

            Parameters:
            --------
                file: str
                    File name of .bin type file
                Lx: int
                    Size of system in x direction
                beg: int
                    Time step that the system begin become steady.
                h: float
                    A value of density at which peak is steep.

        """
        self.FrameSize = Lx * 4
        self.Lx = Lx
        self.beg = beg
        self.end = os.path.getsize(file) // self.FrameSize
        self.file = file
        self.h = h

    def get_serials(self):
        """ Calculate time serials of number and location of peaks."""

        f = open(self.file, "rb")
        f.seek(self.beg * self.FrameSize)
        self.xs = np.zeros(self.end - self.beg, dtype=object)
        for i in range(self.end - self.beg):
            buff = f.read(self.FrameSize)
            rhox = np.array(struct.unpack("%df" % (self.Lx), buff))
            self.xs[i] = locatePeak(rhox, self.Lx, self.h)
        f.close()
        self.num_raw = np.array([x.size for x in self.xs])

    def smooth(self, k=10):
        """ Smooth time serials of number of peaks.

            Parameters:
            --------
                k: int
                    Minimum gap between two valid breaking points.

        """
        m = self.num_raw
        bp = []  # list for breaking points
        dm = []
        for i in range(1, m.size):
            dm_i = m[i] - m[i - 1]
            if dm_i != 0:
                bp.append(i)
                dm.append(dm_i)
                while len(bp) >= 2 and \
                        dm[-2] * dm[-1] < 0 and \
                        bp[-1] - bp[-2] < k:
                    m[bp[-2]:i] = m[i]
                    del bp[-1]
                    del dm[-1]
                    if len(bp) == 1:
                        break
                    else:
                        dm[-1] = m[i] - m[bp[-2]]
                        if dm[-1] == 0:
                            del bp[-1]
                            del dm[-1]
        self.num_smoothed = m

    def segment(self, num, edge_wdt=1000):
        """ Cut time serials of peak numbers into segments.

            Parameters:
            --------
                num: np.ndarray
                    Time serials of peak numbers.
                edge_wdt: int
                    Time serials within half of edge_wdt around
                    breaking point are removed.

            Returns:
            --------
                seg_num: np.ndarray
                    Divide the time serials into sgements by number of peaks.
                seg_idx0: np.ndarray
                    Beginning point of each segement.
                seg_idx1: np.ndarray
                    End point of each segement.
        """
        nb_set = [num[0]]
        end_point = [0]
        for i in range(num.size):
            if num[i] != num[i - 1]:
                end_point.append(i)
                nb_set.append(num[i])
        end_point.append(num.size)
        half_wdt = edge_wdt // 2
        seg_idx0 = []
        seg_idx1 = []
        for i in range(len(nb_set)):
            if i == 0:
                x1 = end_point[i]
                x2 = end_point[i + 1] - half_wdt
            elif i == len(nb_set) - 1:
                x1 = end_point[i] + half_wdt
                x2 = end_point[i + 1]
            else:
                x1 = end_point[i] + half_wdt
                x2 = end_point[i + 1] - half_wdt
            if (x1 < x2):
                seg_idx0.append(x1)
                seg_idx1.append(x2)
        seg_idx0 = np.array(seg_idx0)
        seg_idx1 = np.array(seg_idx1)
        seg_num = np.array([num[t] for t in seg_idx0])
        return seg_num, seg_idx0, seg_idx1

    def cumulate(self, seg_num, seg_idx0, seg_idx1, x_h=180):
        """ Calculate time-averaged rho_x

            Parameters:
            --------
                seg_num: np.ndarray
                    Number of peaks of each segment.
                seg_idx0: np.ndarray
                    First index of each segment.
                seg_idx1: np.ndarray
                    Last index of each segment.
                x_h: int
                    Roll the array of rho_x so that rho_x=h at x=x_h.

            Returns:
            --------
                num_set: np.ndarray
                    Set of number of peak.
                sum_rhox: np.ndarray
                    Sum of rho_x over time, array shape (num_set.size, Lx).
                count_rhox: np.ndarray
                    Count of frames for different number of peak.
        """
        num_set = np.unique(seg_num)
        sum_rhox = {key: np.zeros(self.Lx) for key in num_set}
        count_rhox = {key: 0 for key in num_set}

        f = open(self.file, "rb")

        for i in range(seg_idx0.size):
            num = seg_num[i]
            t1 = seg_idx0[i] + self.beg
            t2 = seg_idx1[i] + self.beg
            f.seek(t1 * self.FrameSize)
            for t in range(t1, t2):
                idx = t - self.beg
                if num == self.num_raw[idx]:
                    buff = f.read(self.FrameSize)
                    rhox_t = np.array(struct.unpack("%df" % (self.Lx), buff))
                    for xp in self.xs[idx]:
                        sum_rhox[num] += np.roll(rhox_t, x_h - int(xp))
                        count_rhox[num] += 1
                else:
                    f.seek(self.FrameSize, 1)

        sum_rhox = np.array([sum_rhox[key] for key in num_set])
        count_rhox = np.array([count_rhox[key] for key in num_set])
        return num_set, sum_rhox, count_rhox


class TimeSerialsPhi:
    """ Time serials of order parameters.

        Variables:
        --------
            self.phi_t: np.ndarray
                Time serials of order parameters.
            self.seg_mean: np.ndarray
                Mean order parameters of each segment.
    """

    def __init__(self, file, beg):
        """ Initialize.

            Parameters:
            --------
                file: str
                    Name of .dat file.
                beg: int
                    Time step that the system begin become steady.
        """
        with open(file) as f:
            lines = f.readlines()
            self.phi_t = np.array([float(i.split("\t")[0]) for i in lines])
            self.end = self.phi_t.size
            self.beg = beg

    def segment(self, seg_idx0, seg_idx1):
        """ Divide serials of phi into segements by peak numbers.

            Parameters:
            --------
                seg_idx0: np.ndarray
                    Starting point of each segment.
                seg_idx1: np.ndarray
                    End point of each segment.
            Returns:
            --------
                seg_phi: np.ndarray
                    Mean order parameter of each segment.
        """
        seg_phi = np.zeros(seg_idx0.size)
        for i in range(seg_idx0.size):
            beg_idx = seg_idx0[i] + self.beg
            end_idx = seg_idx1[i] + self.beg
            seg_phi[i] = np.mean(self.phi_t[beg_idx:end_idx])
        return seg_phi

    def moving_average(self, wdt=100):
        """ Moving average for phi.

            Parameters:
            --------
                wdt: int
                    Size of windows for moving average.

            Returns:
            --------
                beg_mov_ave: int
                    First index for moving average.
                end_mov_ave: int
                    Last index for moving average.
                phi_mov_ave: np.ndarray
                    Order parameters after moving average.
        """
        beg_mov_ave = self.beg + wdt
        end_mov_ave = self.end - wdt
        i = np.arange(beg_mov_ave, end_mov_ave)
        phi_mov_ave = np.array([self.phi_t[j - wdt:j + wdt].mean() for j in i])
        return beg_mov_ave, end_mov_ave, phi_mov_ave


def plot_serials(para: list,
                 t_beg: int,
                 t_end: int,
                 num_raw: np.ndarray,
                 num_smoothed: np.ndarray,
                 seg_num: np.ndarray,
                 seg_idx0: np.ndarray,
                 seg_idx1: np.ndarray,
                 seg_phi: np.ndarray,
                 beg_movAve: np.ndarray,
                 end_movAve: np.ndarray,
                 phi_movAve: np.ndarray):
    """ Plot time serials of peak number and phi."""

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10, 6))
    t = np.arange(t_beg, t_end) * 100
    ax1.plot(t, num_raw)
    ax1.plot(t, num_smoothed)
    for i in range(seg_num.size):
        ax1.plot((seg_idx0[i] + t_beg) * 100, seg_num[i], "o")
        ax1.plot((seg_idx1[i] + t_beg) * 100, seg_num[i], "s")
    ax1.set_ylabel(r"$n_b$")

    ax2.plot(np.arange(beg_movAve, end_movAve) * 100, phi_movAve)
    for i in range(seg_num.size):
        ax2.plot([(seg_idx0[i] + t_beg) * 100, (seg_idx1[i] + t_beg) * 100],
                 [seg_phi[i]] * 2)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\phi$")

    ax1.set_title(r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\, seed=%d$" %
                  (para[0], para[1], para[2], para[3], para[4]))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_rhox_mean(para, num_set, sum_rhox, count_rhox, xlim=None, ylim=None):
    """ Plot time-averaged density profile rho_x.

        Parameters:
        --------
            para: list
                Parameters: eta, eps, Lx, Ly, seed
            num_set: np.ndarray
                Set of number of peaks.
            sum_rhox: np.ndarray
                Sum of rhox over time for each peak number, respectively.
            count_rhox: np.ndarray
                Count of rhox with the same peak number.
            xlim: tuple
                (xmin, xmax)
            ylim: tuple
                (ylim, ymax)
    """

    eta, eps, Lx, Ly, seed = para
    x = np.arange(Lx) + 0.5
    for i in range(num_set.size):
        rhox = sum_rhox[i] / count_rhox[i]
        plt.plot(x, rhox, label=r"$n_b=%d$" % num_set[i])
    plt.title(r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\, seed=%d$" %
              (eta, eps, Lx, Ly, seed))
    plt.legend(loc="best")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
    plt.close()


def handle(eta, eps, Lx, Ly, seed, t_beg=10000, h=1.8, show=False, out=False):
    """ Handle the data for given parameters.

        Parameters:
        --------
            eta: int
            eps: int
            Lx: int
            Ly: int
            Seed: int
            t_beg: int
                Time step that the system become equilibrium.
            h: float
            show: bool
                Whether to show the plot of results.
            out: bool
                Whether to output results.
    """

    def output():
        """ Output time-averaged phi and peak profile for varied num_peak"""
        file = "mb_%d.%d.%d.%d.%d.npz" % (eta, eps, Lx, Ly, seed)
        np.savez(
            file,
            t_beg_end=np.array([t_beg, t_end]),
            num_raw=peak.num_raw,
            num_smoothed=peak.num_smoothed,
            seg_num=seg_num,
            seg_idx0=seg_idx0,
            seg_idx1=seg_idx1,
            seg_phi=seg_phi,
            beg_movAve=beg_movAve,
            end_movAve=end_movAve,
            phi_movAve=phi_movAve,
            num_set=num_set,
            sum_rhox=sum_rhox,
            count_rhox=count_rhox)

    file_phi = "p%d.%d.%d.%d.%d.dat" % (eta, eps, Lx, Ly, seed)
    file_rhox = "rhox_%d.%d.%d.%d.%d.bin" % (eta, eps, Lx, Ly, seed)
    phi = TimeSerialsPhi(file_phi, t_beg)
    peak = TimeSerialsPeak(file_rhox, Lx, t_beg, h)
    t_end = min(phi.end, peak.end)
    phi.end = t_end
    peak.end = t_end
    peak.get_serials()
    peak.smooth()
    seg_num, seg_idx0, seg_idx1 = peak.segment(peak.num_smoothed)
    seg_phi = phi.segment(seg_idx0, seg_idx1)
    beg_movAve, end_movAve, phi_movAve = phi.moving_average()
    num_set, sum_rhox, count_rhox = peak.cumulate(seg_num, seg_idx0, seg_idx1)
    para = [eta / 1000, eps / 1000, Lx, Ly, seed]
    if show:
        plot_serials(para, t_beg, t_end, peak.num_raw, peak.num_smoothed,
                     seg_num, seg_idx0, seg_idx1, seg_phi, beg_movAve,
                     end_movAve, phi_movAve)
        plot_rhox_mean(para, num_set, sum_rhox, count_rhox)
    if out:
        output()


def get_para(file):
    """ Get parameters from filename.

        Parameters:
        --------
            file: str
                Name of input file.

        Returns:
        --------
            para: list
                eta, eps, Lx, Ly, seed
    """
    strList = file.replace("rhox_", "").replace(".bin", "").split(".")
    para = [int(i) for i in strList]
    return para


def all_file():
    """ Handle all files in the target path."""

    files = glob.glob("rhox_*.bin")
    for file in files:
        eta, eps, Lx, Ly, seed = get_para(file)
        handle(eta, eps, Lx, Ly, seed, out=True)
        print("Success for %s" % file)
        # try:
        #     handle(eta, eps, Lx, Ly, seed, out=True)
        #     print("Success for %s" % file)
        # except:
        #     print("Error when handling %s" % file)


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    if len(sys.argv) == 2 and sys.argv[1] == "all":
        all_file()
    elif len(sys.argv) == 6:
        eta = int(sys.argv[1])
        eps = int(sys.argv[2])
        Lx = int(sys.argv[3])
        Ly = int(sys.argv[4])
        seed = int(sys.argv[5])
        handle(eta, eps, Lx, Ly, seed, show=True)
    else:
        print("Error, wrong args!")
        sys.exit()
