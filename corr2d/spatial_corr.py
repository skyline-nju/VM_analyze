"""
Calculate 2D spatial correlation functions of density, current and orientation
fileds.

"""

import numpy as np
import matplotlib.pyplot as plt


def auto_corr2d(f):
    """
    Calculate the auto-correlation of function f(x, y).

    Parameters:
    --------
    f : array_like
        Input array, with shape (n, n).

    Returns:
    --------
    h : array_like
        Unscaled auto-correlation function, where h[0, 0] is np.sum(f**2).
    """
    F = np.fft.rfft2(f)
    h = np.fft.irfft2(F * F.conj())
    return h


def cal_corr2d(num,
               vx,
               vy,
               dA,
               vxm=None,
               vym=None,
               weighted=True,
               remove_mean=False):
    """
    Calculate spatial correlation function of density and velocity.

    Parameters:
    --------
    num : array_like
        Number of particles in each grid.
    vx, vy : array_like
        Mean velocity of particles in each grid.
    dA : float
        Area of one grid.
    vxm, vym : float, optional
        Mean values of vx, vy. If not given, vm=v*num/np.sum(num).
    weighted : bool, optional
        If false, return the correlation function of v, elsewise return the
        correlation function of v * sum, divided by correlation function of
        num.
    remove_mean : bool, optional
        If true, return the correlation function substracted mean values and
        divided by the variance of corresponding field.

    Returns:
    --------
        C_rho, C_v : np.array
            Correlation functions for density and velocity.
    """

    rho = num / dA
    C_rho = np.fft.fftshift(auto_corr2d(rho)) / rho.size
    if weighted:
        C_v = np.fft.fftshift(
            auto_corr2d(vx * rho) + auto_corr2d(vy * rho)) / (C_rho * rho.size)
    else:
        C_v = np.fft.fftshift(auto_corr2d(vx) + auto_corr2d(vy)) / rho.size
    if remove_mean:
        rho_mean = np.mean(rho)
        C_rho -= rho_mean**2
        if vxm is None:
            N = np.sum(num)
            vxm = np.sum(vx * num) / N
            vym = np.sum(vy * num) / N
        C_v -= (vxm * vxm + vym * vym)
        C_rho /= np.var(rho)
        C_v /= C_v[C_v.shape[0] // 2, C_v.shape[1] // 2]
    return C_rho, C_v


def output(r, c_rho, c_v, eta, eps, seed, L, lbox):
    """ Output shperially averaged correlation function into disk. """
    file = "cr_%g_%g_%d_%d_%d.dat" % (eta, eps, L, lbox, seed)
    with open(file, "w") as f:
        lines = [
            "%f\t%.8f\t%.8f\n" % (r[i], c_rho[i], c_v[i])
            for i in range(r.size)
        ]
        f.writelines(lines)


def spherical_average(corr, L, smoothed=True):
    """
    Average correlation function spherically.

    Parameters:
    --------
    corr : array_like
        Input correlation function with dimension two.
    L : int
        Size of the system.
    smoothed : bool, optional
        If true, smooth the averaged correlation function.

    Returns:
    --------
    r : array_like
        The array of radius.
    corr_r : array_like
        Averaged correlation function circularly.
    """
    nRow, nCol = corr.shape
    dict_count = {}
    dict_cr = {}
    rr_max = ((min(nRow, nCol)) // 2)**2
    for row in range(nRow):
        dy = row - nRow // 2
        for col in range(nCol):
            dx = col - nCol // 2
            rr = dx * dx + dy * dy
            if rr < rr_max:
                if rr in dict_count.keys():
                    dict_count[rr] += 1
                    dict_cr[rr] += corr[row, col]
                else:
                    dict_count[rr] = 1
                    dict_cr[rr] = corr[row, col]
    rr_sorted = np.array(sorted(dict_count.keys()))
    r = np.sqrt(rr_sorted) * L / nRow
    corr_r = np.array([dict_cr[key] / dict_count[key] for key in rr_sorted])
    if smoothed:
        r, corr_r = smooth(r, corr_r)
    return r, corr_r


def smooth(x0, y0, x_threshold=10):
    x, y = [], []
    i = 0
    x_t = x_threshold
    while i < x0.size:
        if x0[i] < x_threshold:
            x.append(x0[i])
            y.append(y0[i])
            i += 1
        else:
            j = i + 1
            while j < x0.size and x0[j] < x_t + 1:
                j += 1
            x.append(np.mean(x0[i:j]))
            y.append(np.mean(y0[i:j]))
            i = j
            x_t += 1
    x = np.array(x)
    y = np.array(y)
    return x, y


def get_chara_length(r, c_r, threshold=0.5):
    j = -1
    for i in range(r.size):
        if c_r[i] > threshold and c_r[i + 1] <= threshold:
            j = i
            break
    if j < 0:
        Lc = 0
    else:
        try:
            Lc = r[j] - (c_r[j] - threshold) * (r[j] - r[j + 1]) / (
                c_r[j] - c_r[j + 1])
        except RuntimeError:
            return None
    return Lc


def coarse_grain_array(a, lx=1, ly=1, w=None):
    """
    Merge each lx * ly subcells into new cells.

    Parameters:
    --------
    a : array_like
        Input 2d array.
    lx, ly: int, optional
        Size of boxes to coarse grain the input array.
    w : array_like, optional
        The same shape as f. The weights are used to normalized a. If not
        given, the value for a new cell is the summation over the
        corresponding subcells.

    Returns:
    --------
    res : array_like
        The array to output.
    """
    nrows0, ncols0 = a.shape
    nrows = nrows0 // ly
    ncols = ncols0 // lx
    res = np.zeros((nrows, ncols))
    if w is None:
        a1 = np.array(
            [np.sum(a[i * ly:(i + 1) * ly], axis=0) for i in range(nrows)])
        a2 = np.array(
            [np.sum(a1[:, j * lx:(j + 1) * lx], axis=1) for j in range(ncols)])
        res = a2.T
    else:
        a1 = np.array([
            np.sum(a[i * ly:(i + 1) * ly] * w[i * ly:(i + 1) * ly], axis=0)
            for i in range(nrows)
        ])
        a2 = np.array(
            [np.sum(a1[:, j * lx:(j + 1) * lx], axis=1) for j in range(ncols)])
        norm_factor = coarse_grain_array(w, lx, ly)
        res = a2.T
        mask = norm_factor > 0
        res[mask] /= norm_factor[mask]
    return res


def plot_two_Cv(L, t, num, vx, vy, *args):
    """ Show correlation functions for velocity fields, with and without weighted
        term.

    """
    ly0 = L / num.shape[0]
    lx0 = L / num.shape[1]
    C_rho, C_v = cal_corr2d(num, vx, vy, lx0 * ly0)
    r, cv = spherical_average(C_v, L)
    plt.plot(r, cv, label=r"$l=%g$" % lx0)
    for k in args:
        num2 = coarse_grain_array(num, k, k)
        vx2 = coarse_grain_array(vx, k, k, num)
        vy2 = coarse_grain_array(vy, k, k, num)
        C_rho, C_v = cal_corr2d(num2, vx2, vy2, lx0 * ly0 * k * k)
        r, cv = spherical_average(C_v, L)
        plt.plot(r, cv, label=r"$l=%g$" % (lx0 * k))
    C_rho, C_v = cal_corr2d(num, vx, vy, lx0 * ly0, weighted=False)
    r, cv = spherical_average(C_v, L)
    plt.plot(r, cv, "--", label=r"$l=%g$" % lx0)
    for k in args:
        num2 = coarse_grain_array(num, k, k)
        vx2 = coarse_grain_array(vx, k, k, num)
        vy2 = coarse_grain_array(vy, k, k, num)
        C_rho, C_v = cal_corr2d(
            num2, vx2, vy2, lx0 * ly0 * k * k, weighted=False)
        r, cv = spherical_average(C_v, L)
        plt.plot(r, cv, "--", label=r"$l=%g$" % (lx0 * k))
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim(xmax=60)
    # plt.ylim(ymin=1e-3)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$C_{\rho}(r)$")
    plt.axvline(10)
    plt.legend()
    plt.suptitle(r"$t=%d$" % t)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def vary_box_size(magnification, L, t, num, vx, vy, rm_mean=True):
    """
    Show effects of varied boxes size used for coarse grain on the resulting
    correlation functions. Lef panel for density and right pannel for velocity.

    Parameters:
    --------
    magnification : list
        List of magnification factors to enlarge boxes size.
    L : int
        System size.
    t : int
        Time.
    num : array_like
        Particle number in each cell.
    vx : array_like
        Vx averaged over each cell.
    vy : array_like
        Vy averaged over each cell.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ly0 = L / num.shape[0]
    lx0 = L / num.shape[1]
    C_rho, C_v = cal_corr2d(num, vx, vy, lx0 * ly0, remove_mean=rm_mean)
    r, cr = spherical_average(C_rho, L)
    r, cv = spherical_average(C_v, L)
    ax1.plot(r, cr, label=r"$l=%g$" % lx0)
    ax2.plot(r, cv, label=r"$l=%g$" % lx0)
    for k in magnification:
        num2 = coarse_grain_array(num, k, k)
        vx2 = coarse_grain_array(vx, k, k, num)
        vy2 = coarse_grain_array(vy, k, k, num)
        C_rho, C_v = cal_corr2d(
            num2, vx2, vy2, lx0 * ly0 * k * k, remove_mean=rm_mean)
        r, cr = spherical_average(C_rho, L)
        r, cv = spherical_average(C_v, L)
        ax1.plot(r, cr, label=r"$l=%g$" % (lx0 * k))
        ax2.plot(r, cv, label=r"$l=%g$" % (lx0 * k))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$C_{\rho}(r)$")
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$C_{V}(r)$")
    ax1.legend()
    ax2.legend()
    plt.suptitle(r"$t=%d$" % t)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def time_average(frames,
                 L,
                 eta,
                 eps,
                 ncols,
                 seed,
                 flag_plot=False,
                 ax=None,
                 magni=1,
                 t_equil=500000,
                 save=False):
    """
    Average correlation function over time.

    Parameters:
    --------
    frames : generator
        Each frame contains t, vxm, vym, num, vx, vy
    L : int
        System size.
    eta : float
        Strength of noise.
    eps: float
        Strength of disorder.
    ncols : int
        Number of rows of cells.
    seed : int
        Random seed.
    ax : array of Axes object, optional
        The axes to plot correlation function.
    magni : int
        Magnifications for enlarging cells.
    t_equil : int
        Equilibrium time to reach steady state.
    save : bool
        Whether to save the obtained result to disk.
    """
    cell_area = (L / ncols * magni)**2
    C_rho_sum = np.zeros((ncols // magni, ncols // magni))
    C_v_sum = np.zeros_like(C_rho_sum)
    count = 0
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        print("t=", t)
        if t >= t_equil:
            if magni != 1:
                vx = coarse_grain_array(vx, magni, magni, num)
                vy = coarse_grain_array(vy, magni, magni, num)
                num = coarse_grain_array(num, magni, magni)
            c_rho, c_v = cal_corr2d(num, vx, vy, cell_area)
            C_rho_sum += c_rho
            C_v_sum += c_v
            count += 1
    C_rho_sum /= count
    C_v_sum /= count
    r, cv = spherical_average(C_v_sum, L)
    r, crho = spherical_average(C_rho_sum, L)
    if ax is None and flag_plot:
        fig, ax = plt.subplots(ncols=2, nrows=1)
        flag_plot = True
    else:
        flag_plot = True
        flag_show = False
    if flag_plot:
        ax[0].plot(r, crho)
        ax[1].plot(r, cv)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
    if flag_show and flag_plot:
        plt.show()
        plt.close()
    if save:
        output(r, crho, cv, eta, eps, seed, L, L // ncols * magni)


if __name__ == "__main__":
    import sys
    sys.path.append("../snap")
    import load_snap
    import os
    os.chdir(r"data")
    file = r"ciff_0.18_0_2048_2048_1024_1024_4194304_1.06_123.bin"
    s = file.replace(".bin", "").split("_")
    eta = float(s[1])
    eps = float(s[2])
    L = int(s[3])
    ncols = int(s[5])
    snap = load_snap.CoarseGrainSnap(file)
    frames = snap.gene_frames(beg_idx=60)
    for frame in frames:
        t, vxm, vym, num, vx, vy = frame
        if t == 404:
            plot_two_Cv(L, t, num, vx, vy, 2, 4)
            sys.exit()
        print(t)
