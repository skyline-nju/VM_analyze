import os
import sys
import timeSerials
import common
import glob
import numpy as np
import platform
import matplotlib
if platform.system() is not "Windows":
    matplotlib.use("Agg")


def show_snaps(dt, para):
    """ Show snapshot of rho_x every dt frame. """

    import matplotlib.pyplot as plt

    pat = common.dict2str(para, "eta", "eps", "Lx", "Ly", "seed")
    files = glob.glob("rhox_%s.bin" % pat)
    for file in files:
        print(file)
        eta, eps, Lx, Ly, seed = common.get_para(file)
        x = np.arange(Lx) + 0.5
        peak = timeSerials.TimeSerialsPeak(file, Lx)
        for i in range(peak.end // dt):
            idx = i * dt
            rhox, xPeak = peak.get_one_frame(idx)
            plt.plot(x, rhox)
            plt.axhline(1.8, c="r")
            for xp in xPeak:
                plt.axvline(xp, c="g")
            plt.title(
                r"$\eta=%g,\epsilon=%g,L_x=%d,L_y=%d,seed=%d,n_b=%d,t=%d$" %
                (eta / 1000, eps / 1000, Lx, Ly, seed, xPeak.size, idx * 100))
            if platform.system() is "Windows":
                plt.show()
            else:
                file = "snap_%d.%d.%d.%d.%d.%08d.png" % (eta, eps, Lx, Ly,
                                                         seed, idx * 100)
                plt.savefig(file)
            plt.close()


if __name__ == "__main__":
    path0 = "E:\\data\\random_torque\\bands\\Lx\\snapshot"
    dt = 10000

    if len(sys.argv) == 1:
        os.chdir(path0)
        print(os.getcwd())
        show_snaps(dt)

    elif len(sys.argv) % 2 == 1:
        argv = {
            sys.argv[i]: sys.argv[i + 1]
            for i in range(1, len(sys.argv), 2)
        }
        for key in argv:
            print("%s: %s" % (key, argv[key]))

        if "path" in argv:
            os.chdir(argv["path"])
            del argv["path"]
        else:
            os.chidr(path0)
        print(os.getcwd())

        if "dt" in argv:
            dt = int(argv["dt"])
            del argv["dt"]

        show_snaps(dt, argv)

    else:
        print("Wrong args! Should be")
        print("path $path dt $dt eta $eta eps $eps...")
