import numpy as np
import struct
import matplotlib.pyplot as plt
import os


def read_snap(filename):
    s = os.path.basename(filename).replace(".bin", "").split("_")
    if len(s) == 7:
        Lx = float(s[1])
        Ly = float(s[2])
        eta = float(s[3])
        rho_0 = float(s[4])
        seed = int(s[5])
        t = int(s[6])
        n = int(Lx * Ly * rho_0)
    with open(filename, "rb") as f:
        buf = f.read()
        data = np.array(struct.unpack("%dd" % (n * 4), buf)).reshape(n, 4).T
        x, y, vx, vy = data
    return x, y, vx, vy, Lx, Ly, eta, rho_0, seed, t


if __name__ == "__main__":
    os.chdir("data")
    filename = "snap_100_180_0.35_1.0_2_050000.bin"
    x, y, vx, vy, Lx, Ly, eta, rho_0, seed, t = read_snap(filename)
    fig = plt.subplots(figsize=(4, 8))
    plt.plot(x, y, ".")
    plt.show()
    plt.close()
