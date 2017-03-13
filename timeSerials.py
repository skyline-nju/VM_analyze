import numpy as np
import os
import struct


def get_std_gap(xp: np.ndarray, Lx: int) -> float:
    """ Calculate std of gaps between nearest peeks.

        Parameters:
        --------
            xp: np.ndarray
                Location of peaks.
            Lx: int
                Box size in x direction.

        Returns:
        --------
            std_gap: float
                Std of gaps betwwen nearest peaks.
    """
    if xp.size <= 1:
        std_gap = 0
    else:
        dx = np.array([xp[i] - xp[i - 1] for i in range(xp.size)])
        dx[dx < 0] += Lx
        std_gap = dx.std()
    return std_gap


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


def get_ave_peak(rhox0, Lx, xPeak, xc, interp=True):
    """ Average peaks of rhox.

        Parameters:
        --------
            rhox0: np.ndarray
                Density profile along x, may have multiple peaks.
            Lx: int
                Box size along x.
            xPeak: np.ndarray
                Locations of x where rho_x=h.
            xc: float
                Shift all x in xPeak to xc.
            interp: bool
                Shift with interpolation.

        Returns:
        -------
            ave_peak: np.ndarray
                Average of peaks.

    """

    pass


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

    def __init__(self, file, Lx, beg=10000, h=1.8):
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

    def get_one_frame(self, t):
        """ Get rho_x and number of peaks at t. """

        f = open(self.file, "rb")
        f.seek(t * self.FrameSize)
        buff = f.read(self.FrameSize)
        f.close()
        rhox = np.array(struct.unpack("%df" % (self.Lx), buff))
        xPeak = locatePeak(rhox, self.Lx, self.h)
        return rhox, xPeak

    def smooth(self, k=10):
        """ Smooth time serials of number of peaks.

            Parameters:
            --------
                k: int
                    Minimum gap between two valid breaking points.

        """
        m = self.num_raw.copy()
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

    def segment(self, num, edge_wdt=2000):
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
        sum_std_gap = {key: 0 for key in num_set}
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
                    sum_std_gap[num] += get_std_gap(self.xs[idx], self.Lx)
                else:
                    f.seek(self.FrameSize, 1)

        sum_rhox = np.array([sum_rhox[key] for key in num_set])
        sum_std_gap = np.array([sum_std_gap[key] for key in num_set])
        count_rhox = np.array([count_rhox[key] for key in num_set])
        return num_set, sum_rhox, sum_std_gap, count_rhox


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
