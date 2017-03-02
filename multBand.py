import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import struct
import sys
import os


def readSnap(file):
    f = open(file, "rb")
    buff = f.read()
    f.close()
    n = len(buff) // 12
    data = np.array(struct.unpack('%df' % (3 * n), buff)).reshape(n, 3).T
    x = data[0]
    y = data[1]
    vx = np.cos(data[2])
    vy = np.sin(data[2])
    return x, y, vx, vy


def coarsing(filename, Lx, Ly):
    x, y, vx, vy = readSnap(filename)
    Nx, Ny = 400, 50
    dx, dy = Lx / Nx, Ly / Ny
    X = np.linspace(dx / 2, Lx - dx / 2, Nx)
    Y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    rho = np.zeros((Ny, Nx))
    dx2 = 10
    Nx2 = math.floor(Lx / dx2)
    rhox = np.zeros(Nx2)
    dx2 = Lx / Nx2
    X2 = np.linspace(dx2 / 2, Lx - dx2 / 2, Nx2)
    dA = dx * dy
    n = len(x)
    phix = 0
    phiy = 0
    for k in range(n):
        i = math.floor(x[k] / dx)
        if i == Nx:
            i -= 1
        j = math.floor(y[k] / dy)
        if j == Ny:
            j -= 1
        rho[j][i] += 1
        i2 = math.floor(x[k] / dx2)
        if i2 == Nx2:
            i2 -= 1
        rhox[i2] += 1
        phix += vx[k]
        phiy += vy[k]
    phi = math.sqrt(phix * phix + phiy * phiy) / n
    rho /= dA
    rhox /= (dx2 * Ly)
    return X, Y, X2, rho, rhox, phi


def cal_phi_from_snap(Lx, eta=350, eps=0, Ly=200):
    files = glob.glob("snapshot\\buff\\s%d.%d.%d.%d.*.bin" %
                      (eta, eps, Lx, Ly))
    phi = 0
    count = 0
    for file in files:
        t = int(file.split(".")[-2])
        if t >= 20:
            x, y, vx, vy = readSnap(file)
            vxm, vym = np.mean(vx), np.mean(vy)
            phi += np.sqrt(vxm**2 + vym**2)
            count += 1
    return phi / count


def average_rho_x_(Lx):
    def coarse_x(x, nbins, Lx, Ly=200):
        hist_x, bin_edges = np.histogram(x, nbins, range=(0, Lx))
        rho_x = hist_x / (Lx * Ly) * nbins
        bin_mid = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        return rho_x, bin_mid

    h = 1.8
    sum_rho_x = np.zeros(Lx)
    count = 0
    bin_mid = None
    if Lx >= 400:
        seed = 25200
    else:
        seed = 5000 + Lx
    for t in range(30, 100):
        file = r"snapshot\buff\s350.0.%d.200.%d.%04d.bin" % (Lx, seed, t)
        x, y, vx, vy = readSnap(file)
        rho_x, bin_mid = coarse_x(x, Lx, Lx)
        nPeak, xPeak = locatePeak(rho_x, bin_mid, Lx, h)
        if nPeak != 2:
            plt.plot(bin_mid, rho_x, "r")
            plt.plot(bin_mid, np.roll(rho_x, 180 - int(xPeak[0])), "b")
            plt.plot(bin_mid, np.roll(rho_x, 180 - int(xPeak[1])), "g")

            plt.plot(xPeak, np.ones(len(xPeak)) * h, "s")
            plt.xlim(0, Lx)
            plt.suptitle(r"$t=%d\ n_b=%d$" % (t, nPeak))
            plt.show()
            plt.close()
        else:
            count += 2
            rho_x1 = np.roll(rho_x, 180 - int(xPeak[0]))
            rho_x2 = np.roll(rho_x, 180 - int(xPeak[1]))
            sum_rho_x += rho_x1 + rho_x2
    mean_rho_x = sum_rho_x / count
    return bin_mid, mean_rho_x


def read_rho_x(file, Lx=None):
    if Lx is None:
        Lx = int(file.split(".")[-4])
    rho_x = np.fromfile(file, dtype=np.float32)
    if rho_x.size % Lx == 0:
        nStep = rho_x.size // Lx
        rho_x = rho_x.reshape(nStep, Lx)
        return rho_x
    else:
        print("Error, wrong file size")
        sys.exit(0)


def read_phi(file):
    with open(file) as f:
        lines = f.readlines()
    phi = np.array([float(i.split("\t")[0]) for i in lines])
    return phi


def locatePeak(rho_x, bin_mid, Lx, h=1.8):
    def check_gap(x, dxm):
        # cheak whether the gap between two nearest bands is
        # large than a threshold value dxm
        def loop():
            x_pre = x[0]
            i = len(x) - 1
            while i >= 0:
                dx = x_pre - x[i]
                if dx < 0:
                    dx += Lx
                if dx < dxm:
                    del x[i]
                else:
                    x_pre = x[i]
                i -= 1

        x0 = x.copy()
        loop()
        # in case of x[0] is invalid
        if len(x) == 0 or x[0] != x0[0]:
            x = x0.copy()
            del x[0]
            loop()
        return x

    n = rho_x.size
    xPeak = []
    for i in range(n):
        if rho_x[i - 1] > h and \
           rho_x[i] <= h and \
           rho_x[(i + 10) % n] < h and \
           rho_x[(i + 20) % n] < h and \
           rho_x[(i + 30) % n] < h and \
           rho_x[(i + 40) % n] < h:
            if i == 0:
                x_left = bin_mid[i - 1] - Lx
            else:
                x_left = bin_mid[i - 1]
            x_right = bin_mid[i]
            x = x_left - (rho_x[i - 1] - h) / (rho_x[i - 1] - rho_x[i]) * (
                x_left - x_right)
            if x < 0:
                x += Lx
            xPeak.append(x)

    if len(xPeak) > 2:
        xPeak = check_gap(xPeak, 10)
    if len(xPeak) > 2:
        xPeak = check_gap(xPeak, 100)
    return len(xPeak), xPeak


def peak_serials(Lx, rho_xs=None):
    if rho_xs is None:
        file = r"snapshot\rhox\rhox_350.0.%d.200.%d.bin" % (Lx, 214000 + Lx)
        rho_xs = read_rho_x(file)
    h = 1.8
    nb = np.zeros(rho_xs.shape[0], int)
    xPeak = np.zeros(nb.size, list)
    x = np.arange(Lx) + 0.5
    for i, rho_x in enumerate(rho_xs):
        nb[i], xPeak[i] = locatePeak(rho_x, x, Lx, h)
    plt.plot(np.arange(rho_xs.shape[0]), nb)
    plt.suptitle(r"$L_x=%d$" % (Lx))
    plt.show()
    plt.close()


def nBand_serial(Lx, file=None, seed=None, eta=350, eps=0, h=1.8):
    if file is None and seed is not None:
        file = r"snapshot\rhox\rhox_%d.%d.%d.200.%d.bin" % (eta, eps, Lx, seed)
    rho_x = read_rho_x(file)
    nb = np.zeros(rho_x.shape[0], int)
    xPeak = np.zeros(nb.size, list)
    x = np.arange(Lx) + 0.5
    for i, y in enumerate(rho_x):
        nb[i], xPeak[i] = locatePeak(y, x, Lx, h)
    return nb, xPeak, rho_x


def smooth(h_raw, k=10):
    # eliminate the saw-toothed flucuations (width < k) of h.
    h = h_raw.copy()
    x_turn = []  # list for turning point
    dh = []
    for i in range(1, h.size):
        dh_i = h[i] - h[i - 1]
        if dh_i != 0:
            x_turn.append(i)
            dh.append(dh_i)
            while len(x_turn) >= 2 and \
                    dh[-2] * dh[-1] < 0 and \
                    x_turn[-1] - x_turn[-2] < k:
                h[x_turn[-2]:i] = h[i]
                del x_turn[-1]
                del dh[-1]
                if len(x_turn) == 1:
                    break
                else:
                    dh[-1] = h[i] - h[x_turn[-2]]
                    if dh[-1] == 0:
                        del x_turn[-1]
                        del dh[-1]
    return h


def cut_edge(nb, ncut=5000, width=1000):
    # remove the serials around each breaking point
    nb_set = [nb[ncut]]
    x_turn = [ncut]
    for i in range(ncut + 1, nb.size):
        if nb[i] != nb[i - 1]:
            x_turn.append(i)
            nb_set.append(nb[i])
    x_turn.append(nb.size)
    half_wdt = width // 2
    dict_nb = {}
    for i in range(len(nb_set)):
        if i == 0:
            x1 = x_turn[i]
            x2 = x_turn[i + 1] - half_wdt
        elif i == len(nb_set) - 1:
            x1 = x_turn[i] + half_wdt
            x2 = x_turn[i + 1]
        else:
            x1 = x_turn[i] + half_wdt
            x2 = x_turn[i + 1] - half_wdt
        if (x1 < x2):
            if nb_set[i] in dict_nb:
                dict_nb[nb_set[i]].append([x1, x2])
            else:
                dict_nb[nb_set[i]] = [[x1, x2]]
    return dict_nb


def test_cut_edge(Lx, seed, ncut=5000, eta=350, eps=0, Ly=200):
    def moving_average(x1, x2, y_raw, wdt):
        x = np.arange(x1 + wdt, x2 - wdt)
        y = np.array([np.mean(y_raw[i - wdt:i + wdt]) for i in x])
        return x, y

    file1 = "snapshot\\rhox\\rhox_%d.%d.%d.%d.%d.bin" % (eta, eps, Lx, Ly,
                                                         seed)
    file2 = "snapshot\\rhox\\p%d.%d.%d.%d.%d.dat" % (eta, eps, Lx, Ly, seed)
    nb, xPeak, rho_x = nBand_serial(Lx, file=file1)
    nb = smooth(nb)
    phi = read_phi(file2)
    if nb.size < phi.size:
        phi = phi[:nb.size]
    elif nb.size > phi.size:
        nb = nb[:phi.size]
    x, mean_phi = moving_average(ncut, phi.size, phi, 100)
    plt.plot(x, mean_phi)
    plt.show()
    plt.close()


def accu_rhox(rho_xs, xPeak, t1, t2):
    sum_rhox = np.zeros(rho_xs.shape[1])
    for t in range(t1, t2):
        for x in xPeak[t]:
            sum_rhox += np.roll(rho_xs[t], 180 - int(x))
    return sum_rhox


def fixed_Lx(Lx, Ly=200, eta=350, eps=0):
    # calculate phi and time-averaged peak as functions of Lx and nBand_serial
    files = glob.glob("snapshot\\rhox\\rhox_%d.%d.%d.%d.*.bin" %
                      (eta, eps, Lx, Ly))
    dict_phi = {}
    dict_rhox = {}
    dict_count = {}
    # curmulate
    for file in files:
        seed = int(file.split(".")[-2])
        file_phi = "snapshot\\rhox\\p%d.%d.%d.%d.%d.dat" % (eta, eps, Lx, Ly,
                                                            seed)
        nb, xPeak, rho_x = nBand_serial(Lx, file=file)
        nb = smooth(nb)
        phi = read_phi(file_phi)
        if nb.size < phi.size:
            phi = phi[:nb.size]
        elif nb.size > phi.size:
            nb = nb[:phi.size]
        dict_nb = cut_edge(nb)
        for key in dict_nb.keys():
            for t1, t2 in dict_nb[key]:
                sum_phi = np.sum(phi[t1:t2])
                sum_rhox = accu_rhox(rho_x, xPeak, t1, t2)
                count = t2 - t1
                if key not in dict_phi.keys():
                    dict_phi[key] = sum_phi
                    dict_rhox[key] = sum_rhox
                    dict_count[key] = count
                else:
                    dict_phi[key] += sum_phi
                    dict_rhox[key] += sum_rhox
                    dict_count[key] += count

    tot = sum([dict_count[key] for key in dict_count])
    dict_pb = {}  # probability for nb at Lx
    for key in dict_phi:
        dict_phi[key] /= dict_count[key]  # get mean value of order parameter
        dict_rhox[key] /= (dict_count[key] * key)  # get time-averaged peak
        dict_pb[key] = dict_count[key] / tot
    return dict_phi, dict_pb, dict_rhox


def output_rhox_mean(Lx, eta=350, eps=0, Ly=200, dict_rhox=None):
    print('Lx=%d' % (Lx))
    if dict_rhox is None:
        dict_phi, dict_pb, dict_rhox = fixed_Lx(Lx)
    x = np.arange(Lx) + 0.5
    for nb in dict_rhox.keys():
        lines = ["%.1f\t%f\n" % (x[i], dict_rhox[nb][i]) for i in range(Lx)]
        with open("snapshot\\rhox\\meanPeak_%d.%d.%d.%d_nb%d.dat" %
                  (eta, eps, Lx, Ly, nb), "w") as f:
            f.writelines(lines)


def output_phi(Lxs, nb, eta=350, eps=0, Ly=200):
    L = []
    phi = []
    prob = []
    for Lx in Lxs:
        print("Lx=%d" % (Lx))
        dict_phi, dict_pb, dict_rhox = fixed_Lx(Lx)
        if nb in dict_phi.keys():
            L.append(Lx)
            phi.append(dict_phi[nb])
            prob.append(dict_pb[nb])
    plt.plot(L, phi, "-o")
    plt.show()
    plt.close()
    lines = np.array(
        ["%d\t%f\t%f\n" % (L[i], phi[i], prob[i]) for i in range(len(L))])
    with open("snapshot\\rhox\\meanPhi_%d.%d.%d_nb%d.dat" % (eta, eps, Ly, nb),
              "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    print(os.getcwd())
    Lxs = [280, 300, 320, 340, 360, 380, 400, 420, 440]
    # Lxs = [760]
    # output_phi(Lxs, 2)

    # for Lx in Lxs:
    #     print("Lx=%d" % (Lx))
    #     dict_phi, dict_pb, dict_rhox = fixed_Lx(Lx)
    #     x = np.arange(Lx) + 0.5
    #     if 2 in dict_rhox.keys():
    #         plt.plot(x, dict_rhox[2], label=r"$L_x=%d$" % (Lx))
    #     output_rhox_mean(Lx, dict_rhox=dict_rhox)
    # plt.xlim(80, 200)
    # plt.legend(loc="best")
    # plt.show()
    # plt.close()
    # plt.plot()

    # nb = 2
    # rho_gas = np.zeros(len(Lxs))
    # Nb = np.zeros_like(rho_gas)
    # for i, Lx in enumerate(Lxs):
    #     with open("snapshot\\rhox\\meanPeak_%d.%d.%d.%d_nb%d.dat" % (
    #             350, 0, Lx, 200, nb)) as f:
    #         lines = f.readlines()
    #         rhox = np.array([float(
    #                 line.replace('\n', '').split("\t")[-1]) for line in lines])
    #         rho_gas[i] = np.mean(rhox[190: 220])
    #         Nb[i] = np.sum(rhox[80: 190])
    # with open("snapshot\\rhox\\meanPhi_%d.%d.%d_nb%d.dat" % (
    #         350, 0, 200, nb)) as f:
    #     lines = f.readlines()
    # phi = np.array([float(line.split("\t")[1]) for line in lines])
    # plt.subplot(221)
    # plt.plot(Lxs, Nb, "-ro")
    # plt.plot(Lxs, Nb - rho_gas * Lxs, "-bs")
    # plt.subplot(222)
    # plt.plot(Lxs, Nb / Lxs, "-ro")
    # plt.plot(Lxs, Nb / Lxs - rho_gas, "-bs")
    # plt.subplot(223)
    # plt.plot(Lxs, rho_gas, "-bs")
    # plt.plot(Lxs, 1 - 2 * Nb / Lxs, "-ro")
    # plt.subplot(224)
    # plt.plot(Lxs, phi / (2 * Nb / Lxs), '-s')
    # plt.show()
