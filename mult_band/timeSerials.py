import numpy as np
import os
import struct
from scipy import interpolate


def cal_dx(x1, x2, Lx, dxm=50):
    """ Calculate the displacement of bands.

        Parameters:
        --------
            x1: np.ndarray
                Location of peaks at t1
            x2: np.ndarray
                Location of peaks at t2
            Lx: int
                System size along x direction
            dxm: int
                Max possible displacement

        Returns:
        --------
            dx.mean(): float
                Mean of displacements of each band
    """

    if x1.size != x2.size:
        print("Error, size of x1 should be equal to size of x2")
        return None
    elif x1.size == 0:
        print("Error, sizes of x1 and x2 should be nonzero")
        return None
    else:
        n = x1.size

    if n == 1:
        dx = x2 - x1
    elif 0 < x2[0] - x1[0] < dxm:
        dx = x2 - x1
    elif 0 < x2[0] + Lx - x1[-1] < dxm:
        dx = np.array([x2[i] - x1[i - 1] for i in range(n)])
    else:
        print("Exception when calculating dx")
        print("x1", x1)
        print("x2", x2)
        return None

    dx[dx < 0] += Lx
    return dx.mean()


def cal_v(xs: np.ndarray,
          valid: np.ndarray,
          Lx: int,
          dxm: int = 50,
          dt: int = 100):
    """ Calculate velocity from the time serials of location of bands.

        Parameters:
        --------
            xs: np.ndarray
                Time serials of location of bands
            valid: np.ndarray
                Whether a frame is valid
            Lx: int
                System size in x direction
            dxm: int
                Max possible displacement betwwen two nearest frames
            dt: int
                Time interval of two nearest frames

        Returns:
        --------
            sum_v: float
                Sum of velocity among valid frames
            count: int
                Count of valid frames
    """
    sum_v = 0
    count = 0
    for i in range(1, valid.size):
        if valid[i - 1] and valid[i]:
            dx = cal_dx(xs[i - 1], xs[i], Lx, dxm)
            if dx is not None:
                sum_v += dx / dt
                count += 1
    return sum_v, count


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
    xPeak = check_gap(sorted(xPeak), 100, Lx)
    return np.array(sorted(xPeak))


def get_ave_peak(rhox0, Lx, xPeak, xc, interp=None):
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
            interp: str
                Method of interpolation.

        Returns:
        -------
            ave_peak: np.ndarray
                Average of peaks.

    """
    if xPeak.size > 0:
        mean_rhox = np.zeros(Lx)
        if interp == "nplin":
            x0 = np.arange(Lx) + 0.5
            mean_rhox = np.zeros_like(rhox0)
            for xp in xPeak:
                x_new = x0 - xc + xp
                mean_rhox += np.interp(x_new, x0, rhox0, period=Lx)
        elif interp is None:
            for xp in xPeak:
                mean_rhox += np.roll(rhox0, xc - int(xp))
        elif interp == "linear" or interp == "cubic":
            d = 5
            rhox_tmp = np.zeros(Lx + 2 * d)
            rhox_tmp[:d] = rhox0[Lx - d:]
            rhox_tmp[d:Lx + d] = rhox0
            rhox_tmp[Lx + d:] = rhox0[:d]
            x_tmp = np.arange(-d, Lx + d) + 0.5
            x0 = np.arange(Lx) + 0.5
            f = interpolate.interp1d(x_tmp, rhox_tmp, kind=interp)
            for xp in xPeak:
                x_new = x0 - xc + xp
                x_new[x_new < 0] += Lx
                x_new[x_new >= Lx] -= Lx
                mean_rhox += f(x_new)
        mean_rhox /= xPeak.size
        return mean_rhox
    else:
        return rhox0


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

    def gene_frame(self, beg_idx, end_idx=None, valid=None):
        """ Generator of rhox.

            Parameters:
            --------
                beg_idx: int
                    First frame
                end_idx: int
                    Last frame
                valid: np.ndarray
                    Bool array, invalid frame will be ignored

            Returns:
            --------
                rhox: np.ndarray
                    Density profile along x axis.
                idx: int
                    Index of current frame, return when valid is not None.
        """

        def read_one_frame():
            buff = f.read(self.FrameSize)
            rhox = np.array(struct.unpack("%df" % self.Lx, buff))
            return rhox

        with open(self.file, "rb") as f:
            f.seek(beg_idx * self.FrameSize)
            if end_idx is not None:
                if valid is None:
                    for i in range(beg_idx, end_idx):
                        rhox = read_one_frame()
                        yield rhox
                else:
                    for i, flag in enumerate(valid):
                        if flag:
                            rhox = read_one_frame()
                            idx_cur = i + beg_idx
                            yield rhox, idx_cur
                        else:
                            f.seek(self.FrameSize, 1)
            else:
                rhox = read_one_frame()
                return rhox

    def get_serials(self):
        """ Calculate time serials of number and location of peaks."""

        rhoxs = self.gene_frame(self.beg, self.end)
        self.xs = np.array(
            [locatePeak(rhox, self.Lx, self.h) for rhox in rhoxs])
        self.num_raw = np.array([xs.size for xs in self.xs])
        self.smooth()

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
                if len(nb_set) == 1:
                    x2 = end_point[i + 1]
                else:
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

    def cumulate(self, seg_num, seg_idx0, seg_idx1, x_h=180, interp=None):
        """ Handle the data of time serials of peak and cumulate results.

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
                interp: str
                    Method for interpolation

            Returns:
            --------
                nb_set: np.ndarray
                    Set of number of peak, increasing.
                mean_rhox: np.ndarray
                    Sum of rho_x over time, array shape (num_set.size, Lx).
                std_gap: np.ndarry
                    Mean standard deviation of gaps for varied number of band
                mean_v: np.ndarry
                    Mean velocity of bands for increasing nb
        """
        num_set = np.unique(seg_num)
        sum_rhox = {key: np.zeros(self.Lx) for key in num_set if key > 0}
        sum_std_gap = {key: 0 for key in sum_rhox}
        count_rhox = {key: 0 for key in sum_rhox}
        sum_v = {key: 0 for key in sum_rhox}
        count_v = {key: 0 for key in sum_rhox}

        for i in range(seg_idx0.size):
            nb = seg_num[i]
            if nb == 0:
                continue
            idx0 = seg_idx0[i]
            idx1 = seg_idx1[i]

            # valid[i] == True if xs[i].size == nb
            valid = self.num_raw[idx0:idx1] == nb

            # Calculate mean rhox and standard deviation of gaps betwwen bands.
            gene_rhoxs = self.gene_frame(idx0 + self.beg, idx1 + self.end,
                                         valid)
            for rhox, idx in gene_rhoxs:
                xPeak = self.xs[idx - self.beg]
                sum_rhox[nb] += get_ave_peak(
                    rhox, self.Lx, xPeak, x_h, interp=interp)
                sum_std_gap[nb] += get_std_gap(xPeak, self.Lx)
                count_rhox[nb] += 1

            # Calculate sum of velocity of bands
            sum_v0, count_v0 = cal_v(self.xs[idx0:idx1], valid, self.Lx)
            sum_v[nb] += sum_v0
            count_v[nb] += count_v0

        nb_set = np.array([nb for nb in sorted(sum_rhox.keys())])
        mean_rhox = np.array([sum_rhox[nb] / count_rhox[nb] for nb in nb_set])
        std_gap = np.array([sum_std_gap[nb] / count_rhox[nb] for nb in nb_set])
        mean_v = np.array([sum_v[nb] / count_v[nb] for nb in nb_set])
        return nb_set, mean_rhox, std_gap, mean_v


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # os.chdir(r"E:\data\random_torque\bands\Lx\snapshot\eps0")
    os.chdir(r"D:\tmp")
    eta = 350
    eps = 20
    Lx = 460
    Ly = 200
    seed = 228460
    file = "rhox_%d.%d.%d.%d.%d.bin" % (eta, eps, Lx, Ly, seed)
    peak = TimeSerialsPeak(file, Lx)

    peak.get_serials()
    x = np.arange(Lx) + 0.5
    seg_num, seg_idx0, seg_idx1 = peak.segment(peak.num_smoothed)
    print(seg_num)

    nb_set, rhox2, std_gap, v2 = peak.cumulate(
        seg_num, seg_idx0, seg_idx1, interp="nplin")
    plt.plot(x, rhox2[0], "y")
    plt.axhline(1.8, c="k")
    plt.axvline(180, c="k")
    plt.show()
    print(v2)
