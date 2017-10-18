'''
    Create snapshots to initialize the simulation.
'''

import numpy as np
import matplotlib.pyplot as plt


def create_defect_pair(a, rho, Lx, Ly=None):
    """ Create a pair of defects with separation a.

    Parameters:
    --------
    a : int
        Separation of two defects.
    rho : double
        Density of particles.
    Lx : int
        System size in the x direction.
    Ly : int, optional
        System size in the y direction.
    """
    if Ly is None:
        Ly = Lx
    N = int(rho * Lx * Ly)
    x0, y0 = 0.5 * Lx, 0.5 * Ly
    x, y = np.random.rand(2, N)
    x *= Lx
    y *= Ly
    phi1 = np.arctan2(y-y0, x-x0-a)
    phi2 = np.arctan2(y-y0, x-x0+a)
    phi = phi1 - phi2
    return x, y, phi


def coarse_grain(x, y, phi, Lx, Ly, l=1):
    vx = np.cos(phi)
    vy = np.sin(phi)
    nx = Lx // l
    ny = Ly // l
    vx_new = np.zeros((ny, nx))
    vy_new = np.zeros((ny, nx))
    for k in range(x.size):
        i = int(x[k])
        j = int(y[k])
        vx_new[j, i] += vx[k]
        vy_new[j, i] += vy[k]
    phi_new = np.arctan2(vy_new, vx_new)
    return phi_new


if __name__ == "__main__":
    x, y, phi = create_defect_pair(10, 4, 100)
    phi_grid = coarse_grain(x, y, phi, 100, 100)
    plt.contourf(phi_grid, cmap="hsv")
    plt.colorbar()
    plt.imshow(phi_grid, origin="lower")
    # plt.scatter(x, y, c=phi, cmap="hsv")
    plt.colorbar()
    plt.show()
    plt.close()
