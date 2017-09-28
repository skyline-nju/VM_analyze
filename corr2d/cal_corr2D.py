"""
calculate 2D correlation functions
2017/2/28
"""
import numpy as np
import os
import struct
import glob
import sys


def read_snap(file):
    """
    Read a binary snapshot file and return x, y, theta.

    Parameter:
    --------
    file: string
        Inputted file, cotaining 3 * n float numbers.

    Returns:
    --------
    x: 1d array
        x coordinates with length n.
    y: 1d arry
        y coordinates with length n.
    theta: 1d array
        An serials of angle with length n.

    """
    with open(file, "rb") as f:
        buff = f.read()
    n = len(buff) // 12
    data = struct.unpack("%df" % (3 * n), buff)
    x, y, theta = np.array(data).reshape(n, 3).T
    return x, y, theta


def coarse_grain(file, L, ncols):
    """
    Divide the system into ncols by ncols boxes,
    calculating vx, vy, num of each box.

    """
    x, y, theta = read_snap(file)
    vx = np.zeros((ncols, ncols))
    vy = np.zeros((ncols, ncols))
    num = np.zeros((ncols, ncols), int)
    if ncols == L:
        for i in range(x.size):
            col = int(x[i])
            if col == ncols:
                col = 0
            row = int(y[i])
            if row == ncols:
                row = 0
            vx[row, col] += np.cos(theta[i])
            vy[row, col] += np.sin(theta[i])
            num[row, col] += 1
    else:
        l = L / ncols
        for i in range(x.size):
            col = int(x[i] / l)
            if col == ncols:
                col = 0
            row = int(y[i] / l)
            if row == ncols:
                row = 0
            vx[row, col] += np.cos(theta[i])
            vy[row, col] += np.sin(theta[i])
            num[row, col] += 1
    return vx, vy, num


def corr2d(u, v=None):
    """
    Return a 2d array of correlation function with the same shape as the given
    2D array

    Parameters:
    ----------
    u: 2d array
        Such as 2d field of vx or rho
    v: 2d array
        The same as u.

    Returns:
    ----------
    res: 2d array
        An 2D array of correlation function.

    """

    def autoCorr2d(f):
        F = np.fft.rfft2(f)
        corr = np.fft.irfft2(F * F.conj())
        return corr

    if v is None:
        res = autoCorr2d(u) / u.size
    else:
        res = (autoCorr2d(u) + autoCorr2d(v)) / u.size
    # shift the zero-frequency component to the center of the array.
    res = np.fft.fftshift(res)
    return res


def cal_corr(vx, vy, num, dA):
    """
    Calculate 2D correlation of density, current and orientation.
    The system is divided into equal-size boxes with area dA.

    Parameters:
    --------
    vx: 2d array
        Summation of vx in each box.
    vy: 2d array
        Summation of vy in each box.
    num: 2d array
        Summation of number of particles in each box.
    dA: double
        Area of a box.

    Return:
    --------
    corr_rho: 2d array
        2D correlation function of density.
    corr_J: 2d array
        2D correlation function of current.
    corr_u: 2d array
        2D correlation function of orientation.

    """
    mask = num > 0
    Jx = vx / dA
    Jy = vy / dA
    rho = num / dA
    module = np.zeros_like(Jx)
    module[mask] = np.sqrt(Jx[mask]**2 + Jy[mask]**2)
    ux = np.zeros_like(Jx)
    uy = np.zeros_like(Jy)
    ux[mask] = Jx[mask] / module[mask]
    uy[mask] = Jy[mask] / module[mask]
    valid_count = corr2d(mask)
    corr_rho = corr2d(rho)
    corr_J = corr2d(Jx, Jy) / valid_count
    corr_u = corr2d(ux, uy) / valid_count
    return corr_rho, corr_J, corr_u


class Corr2D:
    def __init__(self, L, ncols, eta, eps, rho0, seed=None):
        self.L = L
        self.ncols = ncols
        self.eta = eta
        self.eps = eps
        self.rho0 = rho0
        self.seed = seed
        self.dA = (L / ncols)**2
        self.count = 0
        self.C_rho = np.zeros((ncols, ncols))
        self.C_J = np.zeros((ncols, ncols))
        self.C_u = np.zeros((ncols, ncols))
        self.sum_vx = 0
        self.sum_vy = 0

    def accu(self, vx, vy, num):
        corr_rho, corr_J, corr_u = cal_corr(vx, vy, num, self.dA)
        self.C_rho += corr_rho
        self.C_J += corr_J
        self.C_u += corr_u
        self.sum_vx += np.mean(vx)
        self.sum_vy += np.mean(vy)
        self.count += 1

    def output(self):
        if not os.path.exists("corr"):
            os.mkdir("corr")
        if self.seed is None:
            file = "corr/corr_%d_%g_%g_%g_%d.npz" % (
                self.L, self.eta, self.eps, self.rho0, self.count)
        else:
            file = "corr/corr_%d_%g_%g_%g_%d_%d.npz" % (
                self.L, self.eta, self.eps, self.rho0, self.seed, self.count)
        if os.path.exists(file):
            print(file, "already exists")
            file = file.replace(".npz", "(2).npz")
        np.savez(
            file,
            C_rho=self.C_rho,
            C_J=self.C_J,
            C_u=self.C_u,
            sum_vx=self.sum_vx,
            sum_vy=self.sum_vy,
            count=self.count)


def cumulate(L, eta, eps, t_beg=400000):
    """
    Output sample-averaged correlation functions.

    Parameters:
    -------
    L: int
        System size
    eta: float
        Strength of noise
    eps: float
        strength of disorder
    t_beg: int
        The equilibrious time step

    """
    rho_0 = 1
    if L >= 512:
        ncols = 512
    else:
        ncols = L
    corr2d = Corr2D(L, ncols, eta, eps, rho_0)
    files = glob.glob("snap/s_%d_%.3f_%.3f_*.bin" % (L, eta, eps))
    print("%d files are found for L = %d, eta = %g, eps = %g, t >= %d" %
          (len(files), L, eta, eps, t_beg))
    if len(files) == 0:
        sys.exit()

    for i, file in enumerate(files):
        t = int(file.replace(".bin", "").split("_")[-1])
        if t >= t_beg:
            vx, vy, num = coarse_grain(file, L, ncols)
            corr2d.accu(vx, vy, num)
        print("%d/%d" % (i, len(files)))
    corr2d.output()


if __name__ == "__main__":
    print(os.getcwd())
    if len(sys.argv) < 5:
        print("need 4 or 5 arguments: paht, L, eta, eps, [t_beg]")
        sys.exit()
    path = sys.argv[1]
    L = int(sys.argv[2])
    eta = float(sys.argv[3])
    eps = float(sys.argv[4])

    os.chdir(path)
    if len(sys.argv) == 6:
        t_beg = sys.argv[5]
        cumulate(L, eta, eps, t_beg)
    else:
        cumulate(L, eta, eps)
