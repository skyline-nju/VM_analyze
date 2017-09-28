import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import colors
import glob
import os
import sys


def spatial_average(corr, L):
    """
    Average correlation function along circles

    Parameters:
    --------
    corr: 2d array
        2d correlation function.
    L: int
        Size of the system.

    Returns:
    --------
    r: 1d array
        1d array of radius.
    corr_r: 1d array
        Averaged correlation function along the radial direction.

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
    return r, corr_r


def read_single(file=None, **para):
    """
    Read spatial correlation functions from a .npz file.
    If file is None, return normed values.

    Parameters:
    --------
    file: string
        .npz file
    para: dict
        keywords: L, eta, eps, rho0, count, seed

    Return:
    --------
    C_rho: 2d array
        Sum of correlation functions for density field
    C_J: 2d array
        Sum of correlation functions for current field,
        where J is the sum of velocity of unit area.
    C_u: 2d array
        Sum of correlations function for orientation filed
    sum_vx: float
        sum of vx for all particles
    sum_vy: float
        sum of vy for all particles
    count: int
        Total number of addends

    """
    flag_norm = False
    if file is None:
        flag_norm = True
        eta = para["eta"]
        eps = para["eps"]
        L = para["L"]
        rho0 = para["rho0"]
        count = para["count"]
        if "seed" in para:
            seed = para["seed"]
            file = "corr_%d_%g_%g_%g_%d_%d.npz" % (L, eta, eps, rho0, seed,
                                                   count)
        else:
            file = "corr_%d_%g_%g_%g_%d.npz" % (L, eta, eps, rho0, count)
    npzfile = np.load(file)
    C_rho = npzfile["C_rho"]
    C_J = npzfile["C_J"]
    C_u = npzfile["C_u"]
    sum_vx = npzfile["sum_vx"]
    sum_vy = npzfile["sum_vy"]
    count = npzfile["count"]
    if flag_norm:
        C_rho /= count
        C_J /= count
        C_u /= count
        theta = np.arctan2(sum_vy, sum_vx)
        return C_rho, C_J, C_u, theta
    else:
        return C_rho, C_J, C_u, sum_vx, sum_vy, count


def read_mult(eta, eps, L, rho0=1.0):
    """
    Read correlation functions from multiple files and return averaged values.

    Parameters:
    -------
    eta: float
    eps: float
    L: int
    rho0: float

    Returns:
    -------
    C_rho: 2d array
        Correlation functions for density field
    C_J: 2d array
        Correlation functions for current field
    C_u: 2d array
        Correlation functions for orientation field
    theta: float
        Averaged polar angle

    """
    files = glob.glob("corr_%d_%g_%g_*.npz" % (L, eta, eps))
    if len(files) == 0:
        print("Error, no matched files for eta=%g, eps=%g, rho_0=%g and L=%d" %
              (eta, eps, rho0, L))
        sys.exit()
    else:
        C_rho, C_J, C_u, sum_vx, sum_vy, count = read_single(files[0])
        for file in files[1:]:
            crho, cj, cu, svx, svy, num = read_single(file)
            C_rho += crho
            C_J += cj
            C_u += cu
            sum_vx += svx
            sum_vy += svy
            count += num
        C_rho /= count
        C_J /= count
        C_u /= count
        theta = np.arctan2(sum_vy, sum_vx)
        return C_rho, C_J, C_u, theta


def single_plot(corr0, theta, L0, nlevel=15, ratio=1, ax=None):
    flag_show = False
    if ax is None:
        ax = plt.subplot(111)
        flag_show = True
    nrows0, ncols0 = corr0.shape
    nrows = int(nrows0 * ratio)
    ncols = int(ncols0 * ratio)
    corr = corr0[nrows0 // 2 - nrows // 2:nrows0 // 2 + nrows // 2, ncols0 // 2
                 - ncols // 2:ncols0 // 2 + ncols // 2]
    L = L0 * ratio
    x = np.linspace(L / ncols * 0.5, L - L / ncols * 0.5, ncols) - L / 2
    y = np.linspace(L / nrows * 0.5, L - L / nrows * 0.5, nrows) - L / 2
    vmin = corr.min()
    vmax = corr0[nrows0 // 2, ncols0 // 2 + 5]
    level = np.linspace(vmin, vmax, nlevel)
    ax.axis("scaled")
    contour = ax.contourf(
        x, y, corr, level, extend="max", norm=colors.PowerNorm(0.7))
    ax.set_xticks(np.linspace(-L / 2, L / 2, 5))
    # ax.set_xticklabels([])
    ax.set_yticks(np.linspace(-L / 2, L / 2, 5))
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$y$")

    Lax = L / 4 * np.cos(theta)
    Lay = L / 4 * np.sin(theta)
    arr_width = 1.5 * ratio * L0 / 2048
    ax.arrow(0, 0, Lax, Lay, width=arr_width, color="r")
    ax.arrow(0, 0, Lay, -Lax, width=arr_width, color="k")
    ax.arrow(-L / 8 * 3.5, -L / 8 * 3.5, L / 8, 0, width=arr_width, color="w")
    ax.arrow(-L / 8 * 3.5, -L / 8 * 3.5, 0, L / 8, width=arr_width, color="w")
    ax.text(
        -L / 8 * 2.5, -L / 8 * 3.5 + 50 * ratio * L0 / 2048, r"$x$", color="w")
    ax.text(
        -L / 8 * 3.5 + 50 * ratio * L0 / 2048, -L / 8 * 2.5, r"$y$", color="w")
    ax.text(
        Lax,
        Lay - 100 * ratio * L0 / 1024,
        r"$v_{\parallel}$",
        color="r",
        fontsize=18)
    ax.text(
        Lay,
        -Lax - 100 * ratio * L0 / 1024,
        r"$v_{\perp}$",
        color="k",
        fontsize=18)

    if flag_show:
        plt.colorbar(contour, ax=ax)
        plt.show()
        plt.close()
    else:
        return ax


def corr_theta(corr, theta, L):
    rowM, colM = corr.shape
    X = np.arange(rowM) - rowM // 2
    Y = np.arange(colM) - colM // 2
    f = interpolate.RectBivariateSpline(X, Y, corr.T)
    r = np.arange(rowM // 2)
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return r * L / rowM, f(x, y, grid=False)


if __name__ == "__main__":
    # change directory to the folder containing targeted .npz files.
    # os.chdir("buff")
    print(os.getcwd())
    L = 724
    eta = 0.1
    eps = 0.03
    C_rho, C_J, C_u, theta = read_mult(eta, eps, L)
    r, cu_rho = spatial_average(C_u, L)
    plt.loglog(r, cu_rho)
    plt.show()
    plt.close()
    single_plot(C_u, theta, L)
