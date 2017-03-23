# calculate velocity of band

import numpy as np


def untangle(x_t: np.ndarray, Lx: int, dxm: int=50):
    """ Remove the periodic boundary condition of time serials of positions.

        N objects moves along x direction in order. Due to the periodic
        boundary condition, the curves of x against time is tangled, which
        can be untangled by this function. The order of objects must be
        unchanged.

        Parameters:
        --------
        x_t: np.ndarray((nt, nx))
            Time serials of positions.
        Lx: int
            Lim of x.
        dxm: int, optional
            Max displacement of one particle betwwen two frame.

        Returns:
        --------
        new_x_t: np.ndarray((nt_new, nx))
            Untangled time serials of positions.
    """
    nt = x_t.shape[0]
    nx = x_t.shape[1]
    new_x_t = [x_t[0]]
    x_pre = x_t[0]
    t = 1
    idx_x0 = 0  # index of first particle
    if isinstance(Lx, int):
        phase_pos = np.zeros(nx, int)
    else:
        phase_pos = np.zeros(nx)
    while t < nt:
        x_cur = x_t[t]
        if 0 < x_cur[0] - x_pre[0] < dxm:
            new_x_t.append(np.roll(x_cur, -idx_x0) + phase_pos)
        elif 0 < x_cur[0] + Lx - x_pre[-1] < dxm:
            idx_x0 += 1
            if idx_x0 >= nx:
                idx_x0 -= nx
            phase_pos[-idx_x0] += Lx
            new_x_t.append(np.roll(x_cur, -idx_x0) + phase_pos)
        else:
            print("error")
            break
        x_pre = x_cur
        t += 1
    new_x_t = np.array(new_x_t)
    return new_x_t


def test_untangle():
    """ Test of function untangle. """
    import matplotlib.pyplot as plt
    Lx = 100
    dxm = 20
    x0 = np.array([0, 30, 60])
    n = x0.size
    x_t = []
    t = np.arange(100)
    for i in t:
        x1 = x0 + 4 * (np.random.rand(n) - 0.5) + 5
        x1[x1 > Lx] -= Lx
        x1.sort()
        x_t.append(x1)
        x0 = x1
    x_t = np.array(x_t)
    print("shape of x_t", x_t.shape)
    plt.subplot(211)
    plt.plot(t, x_t)

    new_x_t = untangle(x_t, Lx, dxm)
    print("shape of new x_t", new_x_t.shape)

    plt.subplot(212)
    plt.plot(t[:new_x_t.shape[0]], new_x_t)
    plt.show()


if __name__ == "__main__":
    test_untangle()
