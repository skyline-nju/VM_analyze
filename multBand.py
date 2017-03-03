import numpy as np
import matplotlib.pyplot as plt
import struct
import os

# def average_rho_x_(Lx):
#     def coarse_x(x, nbins, Lx, Ly=200):
#         hist_x, bin_edges = np.histogram(x, nbins, range=(0, Lx))
#         rho_x = hist_x / (Lx * Ly) * nbins
#         bin_mid = (bin_edges[:-1] + bin_edges[1:]) * 0.5
#         return rho_x, bin_mid

#     h = 1.8
#     sum_rho_x = np.zeros(Lx)
#     count = 0
#     bin_mid = None
#     if Lx >= 400:
#         seed = 25200
#     else:
#         seed = 5000 + Lx
#     for t in range(30, 100):
#         file = r"snapshot\buff\s350.0.%d.200.%d.%04d.bin" % (Lx, seed, t)
#         x, y, vx, vy = readSnap(file)
#         rho_x, bin_mid = coarse_x(x, Lx, Lx)
#         nPeak, xPeak = locatePeak(rho_x, bin_mid, Lx, h)
#         if nPeak != 2:
#             plt.plot(bin_mid, rho_x, "r")
#             plt.plot(bin_mid, np.roll(rho_x, 180 - int(xPeak[0])), "b")
#             plt.plot(bin_mid, np.roll(rho_x, 180 - int(xPeak[1])), "g")

#             plt.plot(xPeak, np.ones(len(xPeak)) * h, "s")
#             plt.xlim(0, Lx)
#             plt.suptitle(r"$t=%d\ n_b=%d$" % (t, nPeak))
#             plt.show()
#             plt.close()
#         else:
#             count += 2
#             rho_x1 = np.roll(rho_x, 180 - int(xPeak[0]))
#             rho_x2 = np.roll(rho_x, 180 - int(xPeak[1]))
#             sum_rho_x += rho_x1 + rho_x2
#     mean_rho_x = sum_rho_x / count
#     return bin_mid, mean_rho_x


def read_phi(file, ncut=0):
    with open(file) as f:
        lines = f.readlines()[ncut:]
    phi = np.array([float(i.split("\t")[0]) for i in lines])
    return phi


def check_gap(xs, dx_min, Lx):
    """
        Check whether dx = xs[i] - xs[i-1] is larger than dx_min for
        i in range(len(xs))
        !!! Remain to be improved

        Parameters:
        --------
        xs: list
            The list to be checked, note that
            0 < xs[0] < xs[1] < ... < xs[n-2] < xs[n-1] < Lx
        dx_min: float
            The minimum distance between two nearest xs
        Lx: float
            The superior limit of x in xs

        Returns:
        --------
        xs: list
            Modified list

    """

    def loop():
        x_pre = xs[0]
        i = len(xs) - 1
        while i >= 0:
            dx = x_pre - xs[i]
            if dx < 0:
                dx += Lx
            if dx < dx_min:
                del xs[i]
            else:
                x_pre = xs[i]
            i -= 1

    xs0 = xs.copy()
    loop()
    # in case of invalid x[0]
    if len(xs) == 0 or xs[0] != xs0[0]:
        xs = xs0.copy()
        del xs[0]
        loop()
    return xs


class Profile_x:
    def __init__(self, Lx, h=1.8, x_h=180):
        self.Lx = Lx
        self.h = h
        self.x_h = x_h
        self.x = np.arange(Lx) + 0.5

    def countPeak(self, rho_x):
        """
            Count number of peaks and find x where rho_x=h

            Parameters:
            --------
            rho_x: 1d array
                Density profile as a function of x

            Return:
            --------
            xPeak: 1d array
                Array of x where rho_x=h

        """
        n = rho_x.size
        xPeak = []
        for i in range(n):
            if rho_x[i - 1] > self.h and \
                    rho_x[i] <= self.h and \
                    rho_x[(i + 10) % n] < self.h and \
                    rho_x[(i + 20) % n] < self.h and \
                    rho_x[(i + 30) % n] < self.h and \
                    rho_x[(i + 40) % n] < self.h:
                if i == 0:
                    x_left = self.x[n - 1] - self.Lx
                else:
                    x_left = self.x[i - 1]
                x_right = self.x[i]
                x = x_left - (rho_x[i - 1] - self.h) / (rho_x[i - 1] - rho_x[i]
                                                        ) * (x_left - x_right)
                if x < 0:
                    x += self.Lx
                xPeak.append(x)
        if len(xPeak) > 2:
            xPeak = check_gap(xPeak, 10, self.Lx)
        if len(xPeak) > 2:
            xPeak = check_gap(xPeak, 100, self.Lx)
        return np.array(xPeak)

    def averagePeak(self, rho_x, xPeak):
        """
            Average the peaks of input density profile,
            roll the density profile so that rhox=h at x_h

            Parameters:
            --------
            rho_x: 1d array
                Input density profile
            xPeak: 1d array
                The array of x where rho_x=h

            Returns:
            --------
            mean_rhox: 1d array
                Averaged density profile over all bands.
        """
        sum_rhox = np.zeros_like(rho_x)
        for x in xPeak:
            sum_rhox += np.roll(rho_x, self.x_h - int(x))
        mean_rhox = sum_rhox / xPeak.size
        return mean_rhox


class TimeSerialsPeak:
    def __init__(self, xPeaks, beg, end, show=False):
        self.x = xPeaks
        self.num = np.array([x.size for x in xPeaks])
        self.beg = beg
        self.end = end
        if show:
            t = np.arange(self.beg, self.end) * 100
            plt.plot(t, self.num)
        self.smooth()
        if show:
            plt.plot(t, self.num)
        self.segment()
        if show:
            for i in range(self.n_lin_seg.size):
                plt.plot(self.t_lin_seg[i], [self.n_lin_seg[i]] * 2, "o")
            plt.show()
            plt.close()

    def smooth(self, k=10):
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
        self.num = m

    def segment(self, edge_wdt=500):
        nb_set = [self.num[0]]
        end_point = [0]
        for i in range(self.num.size):
            if self.num[i] != self.num[i - 1]:
                end_point.append(i)
                nb_set.append(self.num[i])
        end_point.append(self.num.size)
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
        self.n_lin_seg = np.array([self.num[t[0]] for t in lin_seg])
        self.t_lin_seg = (np.array(lin_seg) + self.beg) * 100
        print(self.n_lin_seg)
        print(self.t_lin_seg)


class TimeSerialsPhi:
    def __init__(self, phi, beg, end):
        self.ss = phi
        self.beg = beg
        self.end = end
    
    def moving_average(self, wdt=100):
        i = np.arange(self.beg+wdt, self.end-wdt)
        phi = np.array([np.mean(self.ss[i-wdt: i+wdt])])
        t = i * 100
        return t, phi


class TimeSerials:
    def __init__(self, eta, eps, Lx, Ly, seed, beg_frame=10000, h=1.8,
                 x_h=180):
        self.eta = eta
        self.eps = eps
        self.Lx = Lx
        self.Ly = Ly
        self.seed = seed
        self.beg_frame = beg_frame
        self.phi = read_phi("p%d.%d.%d.%d.%d.dat" %
                            (self.eta, self.eps, self.Lx, self.Ly, self.seed))
        file = "rhox_%d.%d.%d.%d.%d.bin" % (self.eta, self.eps, self.Lx,
                                            self.Ly, self.seed)
        self.FRAME_SIZE = self.Lx * 4
        self.end_frame = min(self.phi.size,
                             os.path.getsize(file) // self.FRAME_SIZE)
        self.tot_frames = self.end_frame - self.beg_frame
        print("Frames: begin=%d, end=%d, total=%d" %
              (self.beg_frame, self.end_frame, self.tot_frames))
        self.phi = self.phi[self.beg_frame:self.end_frame]
        xPeaks = np.zeros(self.tot_frames, dtype=object)
        self.profile_x = Profile_x(Lx, h, x_h)
        self.fin = open(file, "rb")
        self.fin.seek(self.beg_frame * self.FRAME_SIZE)
        for i in range(self.tot_frames):
            rhox = self.read_frames()
            xPeaks[i] = self.profile_x.countPeak(rhox)
        self.fin.close()
        self.peak = TimeSerialsPeak(
            xPeaks, self.beg_frame, self.end_frame, show=True)

    def read_frames(self, n=1):
        buff = self.fin.read(n * self.FRAME_SIZE)
        data = np.array(struct.unpack("%df" % (n * self.Lx), buff))
        if n > 1:
            data = data.reshape(n, self.FRAME_SIZE)
        return data


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    print(os.getcwd())
    eta = 350
    eps = 0
    Lx = 280
    Ly = 200
    seed = 214280
    tss = TimeSerials(eta, eps, Lx, Ly, seed)
