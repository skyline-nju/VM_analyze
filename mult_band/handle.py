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
                 show=False):
    """ Plot time serials of peak number and phi."""

    import matplotlib.pyplot as plt

    eta, eps, Lx, Ly, seed = para
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10, 4))
    t = np.arange(t_beg, t_end) * 100
    ax1.plot(t, num_raw, "y-", label="raw")
    ax1.plot(t, num_smoothed, "r--", label="smoothed")

    tBeg = (seg_idx0 + t_beg) * 100
    tEnd = (seg_idx1 + t_beg) * 100
    ax2.plot(
        np.arange(beg_movAve, end_movAve) * 100,
        phi_movAve,
        "-",
        label="moving average")
    for i in range(seg_num.size):
        line, = ax2.plot([tBeg[i], tEnd[i]], [seg_phi[i]] * 2, "k-", lw=2)
    line.set_label("averaged over valid region")

    # add vertical span on valid regions
    for i in range(seg_num.size):
        if i == 0:
            beg = 0
            end = tBeg[i]
        else:
            beg = tEnd[i-1]
            end = tBeg[i]
        ax1.axvspan(beg, end, alpha=0.2)
        ax2.axvspan(beg, end, alpha=0.2)

    # set label and title
    ax1.text(0.01, 0.9, r"$n_b$", transform=ax1.transAxes, fontsize="x-large")
    ax2.text(0.01, 0.9, r"$\phi$", transform=ax2.transAxes, fontsize="x-large")
    ax1.text(0.985, 0.01, r"$t$", transform=ax1.transAxes, fontsize="x-large")
    ax2.text(0.985, 0.01, r"$t$", transform=ax2.transAxes, fontsize="x-large")
    bbox = dict(edgecolor="k", fill=False)
    ax1.text(0.972, 0.87, "$(a)$", transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.97, 0.875, "$(b)$", transform=ax2.transAxes, bbox=bbox)

    ax1.set_title(
        r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\, {\rm seed}=%d$" %
        (eta / 1000, eps / 1000, Lx, Ly, seed),
        color="b",
        fontsize="large")

    # set lims of x, y axis
    ax1.set_xlim(1e6, 1.2e7)
    ax1.set_ylim(3.8, 6.2)

    plt.tight_layout()
    if show:
        ax1.legend(loc=(0.01, 0.02), fontsize="large")
        ax2.legend(loc=(0.45, 0.02), fontsize="large")
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
            nb_set=nb_set,
            mean_rhox=mean_rhox,
            std_gap=std_gap,
            mean_v=mean_v)
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
    seg_num, seg_idx0, seg_idx1 = peak.segment(peak.num_smoothed)
    seg_phi = phi.segment(seg_idx0, seg_idx1)
    beg_movAve, end_movAve, phi_movAve = phi.moving_average()
    nb_set, mean_rhox, std_gap, mean_v = peak.cumulate(
        seg_num, seg_idx0, seg_idx1, interp=interp)
    para = [eta, eps, Lx, Ly, seed]
    plot_serials(para, t_beg, t_end, peak.num_raw, peak.num_smoothed, seg_num,
                 seg_idx0, seg_idx1, seg_phi, beg_movAve, end_movAve,
                 phi_movAve)
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
    if len(sys.argv) % 2 == 1:
        argv = {
            sys.argv[i]: sys.argv[i + 1]
            for i in range(1, len(sys.argv), 2)
        }
        for key in argv:
            print("%s: %s" % (key, argv[key]))

        if "path" in argv:
            os.chdir(argv["path"])
            del argv["path"]
        print(os.getcwd())

        if "t_beg" in argv:
            t_beg = int(argv["t_beg"])
            del argv["t_beg"]
        else:
            t_beg = 10000

        handle_files(argv, t_beg, show=False, interp="nplin")

    else:
        print("Wrong args! Should be")
        print("path $path t_beg $t_beg eta $eta eps $eps...")
