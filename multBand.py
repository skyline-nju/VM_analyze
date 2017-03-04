import numpy as np
import matplotlib.pyplot as plt
import struct
import os


def read_phi(file, ncut=0):
    with open(file) as f:
        lines = f.readlines()[ncut:]
    phi = np.array([float(i.split("\t")[0]) for i in lines])
    return phi


def check_gap(xs, dx_min, Lx):
    """
        Check whether dx = xs[i] - xs[i-1] is larger than dx_min for
        i in range(len(xs))

        Args:
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


class Profile_x:
    # Density profile along x axis, bin_size=1
    def __init__(self, Lx, h=1.8, x_h=180):
        """
            Parameters:
            -------
            Lx: int
                Size of rho_x
            h: float
                The peak at rho_x=h should be very steep
            x_h: float
                Roll the rho_x so that rho_x=h at x=x_h,
                see function averagePeak
        """
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

        xPeak = check_gap(xPeak, 10, self.Lx)
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
    # Time serials of number and location of peaks
    def __init__(self, xPeaks, beg, end, ax=None):
        self.x = xPeaks
        self.num = np.array([x.size for x in xPeaks])
        if ax is not None:
            t = np.arange(beg, end) * 100
            ax.plot(t, self.num)
        self.smooth()
        if ax is not None:
            ax.plot(t, self.num)
        self.segment(beg)
        if ax is not None:
            for i in range(self.n_lin_seg.size):
                ax.plot(self.t_lin_seg[i] * 100, [self.n_lin_seg[i]] * 2, "o")
            ax.set_ylabel(r"$n_b$")

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

    def segment(self, beg, edge_wdt=1000):
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
        self.t_lin_seg = np.array(lin_seg) + beg


class TimeSerialsPhi:
    def __init__(self, phi, beg, end, t_seg, ax=None):
        self.ss = phi
        self.get_phi_seg(t_seg)
        if ax is not None:
            t, phi = self.moving_average(beg, end)
            ax.plot(t, phi)
            for i, t in enumerate(t_seg):
                ax.plot(t * 100, [self.phi_seg[i]] * 2)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\phi$")

    def get_phi_seg(self, t_seg):
        self.phi_seg = np.array([self.ss[t1:t2].mean() for (t1, t2) in t_seg])

    def moving_average(self, beg, end, wdt=100):
        i = np.arange(beg + wdt, end - wdt)
        phi = np.array([self.ss[j - wdt:j + wdt].mean() for j in i])
        t = i * 100
        return t, phi


class TimeSerials:
    def __init__(self,
                 eta,
                 eps,
                 Lx,
                 Ly,
                 seed,
                 beg_frame=10000,
                 h=1.8,
                 x_h=180,
                 show=False):
        self.eta = eta
        self.eps = eps
        self.Lx = Lx
        self.Ly = Ly
        self.seed = seed
        self.beg_frame = beg_frame
        phi = read_phi("p%d.%d.%d.%d.%d.dat" %
                       (self.eta, self.eps, self.Lx, self.Ly, self.seed))
        file = "rhox_%d.%d.%d.%d.%d.bin" % (self.eta, self.eps, self.Lx,
                                            self.Ly, self.seed)
        self.FRAME_SIZE = self.Lx * 4
        self.end_frame = min(phi.size,
                             os.path.getsize(file) // self.FRAME_SIZE)
        self.tot_frames = self.end_frame - self.beg_frame
        print("Frames: begin=%d, end=%d, total=%d" %
              (self.beg_frame, self.end_frame, self.tot_frames))
        xPeaks = np.zeros(self.tot_frames, dtype=object)
        self.profile_x = Profile_x(Lx, h, x_h)
        self.fin = open(file, "rb")
        self.fin.seek(self.beg_frame * self.FRAME_SIZE)
        for i in range(self.tot_frames):
            rhox = self.read_frames()
            xPeaks[i] = self.profile_x.countPeak(rhox)
        self.fin.close()
        if show:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        else:
            ax1 = None
            ax2 = None
        self.peak = TimeSerialsPeak(
            xPeaks, self.beg_frame, self.end_frame, ax=ax1)
        self.phi = TimeSerialsPhi(
            phi, self.beg_frame, self.end_frame, self.peak.t_lin_seg, ax=ax2)
        if show:
            plt.suptitle(
                r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\ \rm{seed}=%d$" %
                (eta, eps, Lx, Ly, seed))
            plt.show()
            plt.close()

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
    tss = TimeSerials(eta, eps, Lx, Ly, seed, show=True)
