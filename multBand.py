import numpy as np
import matplotlib.pyplot as plt
import struct
import os


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
            rho_x: 1d array
                Density profile as a function of x
            Lx: int
                Length of rho_x
            h: float
                The slope of peak at rho_x=h should be very steep

        Return:
        --------
            xPeak: 1d array
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
    """ class for time serials of peaks."""

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
        """ Calculate time serials of number and location of peaks.

            Variables:
            --------
                self.num: 1d array
                    Time serials of number of peaks.
                self.xs: 2d array
                    Time serials of locations of peaks.
        """
        f = open(self.file, "rb")
        f.seek(self.beg * self.FrameSize)
        self.xs = np.zeros(self.end - self.beg, dtype=object)
        for i in range(self.end - self.beg):
            buff = f.read(self.FrameSize)
            rhox = np.array(struct.unpack("%df" % (self.Lx), buff))
            self.xs[i] = locatePeak(rhox, self.Lx, self.h)
        f.close()
        self.num = np.array([x.size for x in self.xs])

    def smooth(self, k=10):
        """ Smooth time serials of number of peaks.

            Parameters:
            --------
                k: int
                    Minimum gap between two valid breaking points.

            Variables:
            --------
                self.num_s: 1d array
                    Smoothed time serials of peak numbers.
        """
        m = self.num
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
        self.num_s = m

    def segment(self, edge_wdt=1000):
        """ Cut time serials of peak numbers into segments.

            Parameters:
            --------
                edge_wdt: int
                    Time serials within half of edge_wdt around
                    breaking point are removed.

            Variables:
            --------
                self.n_lin_seg: 1d array
                    Number of peaks for each segment.
                self.t_lin_seg: 2d array
                    End points for each segment.
        """
        num = self.num_s
        nb_set = [num[0]]
        end_point = [0]
        for i in range(num.size):
            if num[i] != num[i - 1]:
                end_point.append(i)
                nb_set.append(num[i])
        end_point.append(num.size)
        half_wdt = edge_wdt // 2
        lin_seg = []
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
                lin_seg.append(np.array([x1, x2]))
        self.n_lin_seg = np.array([num[t[0]] for t in lin_seg])
        self.t_lin_seg = np.array(lin_seg) + self.beg

    def cumulate(self, x_h=180):
        """ Calculate time-averaged rho_x

            Parameters:
            --------
                x_h: int
                    Roll the array of rho_x so that rho_x=h at x=x_h.

            Variables:
            --------
                self.num_set: 1d array
                    Set of number of peaks.
                self.sum_rhox: dir
                    Sum of rhox over time for varied peak num.
                self.count: dir
                    Count of each peak number.
                self.mean_rhox: dir
                    Time average of rhox.
        """
        num_set = np.unique(self.n_lin_seg)
        self.sum_rhox = {key: np.zeros(self.Lx) for key in num_set}
        self.count = {key: 0 for key in num_set}

        f = open(self.file, "rb")
        for i, (t1, t2) in enumerate(self.t_lin_seg):
            num = self.n_lin_seg[i]
            f.seek(t1 * self.FrameSize)
            for t in range(t1, t2):
                if num == self.num[t - self.beg]:
                    buff = f.read(self.FrameSize)
                    rhox_t = np.array(struct.unpack("%df" % (self.Lx), buff))
                    for xp in self.xs[t - self.beg]:
                        self.sum_rhox[num] += np.roll(rhox_t, x_h - int(xp))
                        self.count[num] += 1

                else:
                    f.seek(self.FrameSize, 1)
        self.mean_rhox = {
            key: self.sum_rhox[key] / self.count[num]
            for key in num_set
        }
        self.num_set = num_set


class TimeSerialsPhi:
    """ Time serials of order parameters."""

    def __init__(self, file, beg):
        """ Initialize.

            Parameters:
            --------
                file: str
                    Name of .dat file.
                beg: int
                    Time step that the system begin become steady.

            Variables:
            --------
                self.phi_t: 1d array
                    Time serials of order parameters.

        """
        with open(file) as f:
            lines = f.readlines()
            self.phi_t = np.array([float(i.split("\t")[0]) for i in lines])
            self.end = self.phi_t.size
            self.beg = beg

    def get_phi_seg(self, t_seg):
        """ Cut serials of phi into segements according peak numbers.

            Parameters:
            --------
                t_seg: 2d array
                    End points of each segment.

            Variables:
            --------
                self.seg_mean: 1d array
                    Mean order parameters of each segment.
        """
        self.seg_mean = np.array(
            [self.phi_t[t1:t2].mean() for (t1, t2) in t_seg])

    def moving_average(self, wdt=100):
        """ Moving average for phi.

            Parameters:
            --------
                wdt: int
                    Size of windows for moving average.

            Returns:
            --------
                t: 1d array
                    Time serials, the same shape as phi below.
                phi: 1d array
                    Order parameters after moving average.
        """
        i = np.arange(self.beg + wdt, self.end - wdt)
        phi = np.array([self.phi_t[j - wdt:j + wdt].mean() for j in i])
        t = i * 100
        return t, phi


def get_time_serials(file_phi, file_rhox, Lx, beg=10000, h=1.8, show=False):
    """ Get time serials of number and location of peak and order parameters.

        Parameters:
        --------
            file_phi: str
                File name for order parameters.
            file_rhox: str
                File name for rho_x
            Lx: int
                Size of rho_x
            beg: int
                Time step that the system begin become steady.
            h: float
                Used to detect peak of rho_x, the slope of peak
                at rho_x=h is steep.
            show: bool
                Whether to show time serials of peak number and phi.

        Return:
        --------
            peak: TimeSerialsPeak
                Class of peak
            phi: TimeSerialsPhi
                Class of phi
    """
    phi = TimeSerialsPhi(file_phi, beg)
    peak = TimeSerialsPeak(file_rhox, Lx, beg, h)
    end = min(phi.end, peak.end)
    phi.end = end
    peak.end = end
    peak.get_serials()
    peak.smooth()
    peak.segment()
    phi.get_phi_seg(peak.t_lin_seg)
    if show:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        t = np.arange(beg, end) * 100
        ax1.plot(t, peak.num)
        ax1.plot(t, peak.num_s)
        for i in range(peak.n_lin_seg.size):
            ax1.plot(peak.t_lin_seg[i] * 100, [peak.n_lin_seg[i]] * 2, "o")
        ax1.set_ylabel(r"$n_b$")

        t, phi_smothed = phi.moving_average()
        ax2.plot(t, phi_smothed)
        for i, t in enumerate(peak.t_lin_seg):
            ax2.plot(t * 100, [phi.seg_mean[i]] * 2)
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$\phi$")
        plt.show()
        plt.close()
    return peak, phi


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    print(os.getcwd())
    eta = 350
    eps = 20
    Lx = 920
    Ly = 200
    seed = 214920
    # tss = TimeSerials(eta, eps, Lx, Ly, seed, show=True)
    file_phi = "p%d.%d.%d.%d.%d.dat" % (eta, eps, Lx, Ly, seed)
    file_rhox = "rhox_%d.%d.%d.%d.%d.bin" % (eta, eps, Lx, Ly, seed)
    peak, phi = get_time_serials(file_phi, file_rhox, Lx, show=True)
    peak.cumulate()
    plt.plot(peak.mean_rhox[peak.num_set.max()])
    plt.show()
