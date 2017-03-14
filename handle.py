import numpy as np
import timeSerials
import glob
import os
import sys
import platform
import common
import matplotlib
import time
if platform.system() is not "Windows":
    matplotlib.use("Agg")


def plot_serials(para: list,
                 t_beg: int,
                 t_end: int,
                 num_raw: np.ndarray,
                 num_smoothed: np.ndarray,
                 seg_num: np.ndarray,
                 seg_idx0: np.ndarray,
                 seg_idx1: np.ndarray,
                 seg_phi: np.ndarray,
                 beg_movAve: np.ndarray,
                 end_movAve: np.ndarray,
                 phi_movAve: np.ndarray,
                 show=True):
    """ Plot time serials of peak number and phi."""

    import matplotlib.pyplot as plt

    eta, eps, Lx, Ly, seed = para
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10, 6))
    t = np.arange(t_beg, t_end) * 100
    ax1.plot(t, num_raw)
    ax1.plot(t, num_smoothed)
    for i in range(seg_num.size):
        ax1.plot((seg_idx0[i] + t_beg) * 100, seg_num[i], "o")
        ax1.plot((seg_idx1[i] + t_beg) * 100, seg_num[i], "s")
    ax1.set_ylabel(r"$n_b$")

    ax2.plot(np.arange(beg_movAve, end_movAve) * 100, phi_movAve)
    for i in range(seg_num.size):
        ax2.plot([(seg_idx0[i] + t_beg) * 100, (seg_idx1[i] + t_beg) * 100],
                 [seg_phi[i]] * 2)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\phi$")
    ax1.set_title(r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\, seed=%d$" %
                  (eta / 1000, eps / 1000, Lx, Ly, seed))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        outfile = "ts_%d.%d.%d.%d.%d.png" % (eta, eps, Lx, Ly, seed)
        plt.savefig(outfile)
    plt.close()


def handle(eta, eps, Lx, Ly, seed, t_beg=10000, h=1.8, interp=None,
           show=False):
    """ Handle the data for given parameters."""

    def output():
        """ Output time-averaged phi and peak profile for varied num_peak"""
        file = "mb_%d.%d.%d.%d.%d.npz" % (eta, eps, Lx, Ly, seed)
        np.savez(
            file,
            t_beg_end=np.array([t_beg, t_end]),
            num_raw=peak.num_raw,
            num_smoothed=peak.num_smoothed,
            seg_num=seg_num,
            seg_idx0=seg_idx0,
            seg_idx1=seg_idx1,
            seg_phi=seg_phi,
            beg_movAve=beg_movAve,
            end_movAve=end_movAve,
            phi_movAve=phi_movAve,
            num_set=num_set,
            sum_rhox=sum_rhox,
            sum_std_gap=sum_std_gap,
            count_rhox=count_rhox)
        return file

    file_phi = "p%d.%d.%d.%d.%d.dat" % (eta, eps, Lx, Ly, seed)
    file_rhox = "rhox_%d.%d.%d.%d.%d.bin" % (eta, eps, Lx, Ly, seed)
    phi = timeSerials.TimeSerialsPhi(file_phi, t_beg)
    peak = timeSerials.TimeSerialsPeak(file_rhox, Lx, t_beg, h)
    t_end = min(phi.end, peak.end)
    if t_end <= t_beg:
        print("Error, t_beg is larger than t_end!")
        return
    phi.end = t_end
    peak.end = t_end
    peak.get_serials()
    peak.smooth()
    seg_num, seg_idx0, seg_idx1 = peak.segment(peak.num_smoothed)
    seg_phi = phi.segment(seg_idx0, seg_idx1)
    beg_movAve, end_movAve, phi_movAve = phi.moving_average()
    num_set, sum_rhox, sum_std_gap, count_rhox = peak.cumulate(
        seg_num, seg_idx0, seg_idx1, interp=interp)
    para = [eta / 1000, eps / 1000, Lx, Ly, seed]
    plot_serials(
        para,
        t_beg,
        t_end,
        peak.num_raw,
        peak.num_smoothed,
        seg_num,
        seg_idx0,
        seg_idx1,
        seg_phi,
        beg_movAve,
        end_movAve,
        phi_movAve,
        show=show)
    outfile = output()
    return outfile


def handle_files(para_dict, t_beg=10000, out=True, show=False, interp=None):
    """ Handle all matched files. """

    pat = common.dict2str(para_dict, "eta", "eps", "Lx", "Ly", "seed")
    print(pat)
    files = glob.glob("rhox_%s.bin" % pat)
    for i, file in enumerate(files):
        eta, eps, Lx, Ly, seed = common.get_para(file)
        try:
            time_beg = time.clock()
            outfile = handle(
                eta, eps, Lx, Ly, seed, t_beg=t_beg, interp=interp, show=show)
            time_used = time.clock() - time_beg
            print("%d of %d: %s, use time: %g s" %
                  (i + 1, len(files), outfile, time_used))
        except:
            print("Error for %s" % file)


if __name__ == "__main__":
    path0 = "E:\\data\\random_torque\\bands\\Lx\\snapshot"

    if len(sys.argv) == 1:
        os.chdir(path0)
        print(os.getcwd())
        handle_files()

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

        if "t_beg" in argv:
            t_beg = int(argv["t_beg"])
            del argv["t_beg"]
        else:
            t_beg = 5000

        handle_files(argv, t_beg, show=False, interp="cubic")

    else:
        print("Wrong args! Should be")
        print("path $path t_beg $t_beg eta $eta eps $eps...")
