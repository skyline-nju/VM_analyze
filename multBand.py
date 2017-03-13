import numpy as np
import timeSerials
import glob
import os
import sys
import platform
import matplotlib
if platform.system() is not "Windows":
    matplotlib.use("Agg")


def swap_key(dict0):
    """ Swap keys of a nested dict.

        Parameters:
        --------
        dict0: dict
            A dict with two layer keys: {A:{B:object}}

        Returns:
        dict1: dict
            A dict with two layer keys: {B:{A:object}}
    """
    dict1 = {}
    for key1 in dict0:
        for key2 in dict0[key1]:
            if key2 in dict1:
                dict1[key2][key1] = dict0[key1][key2]
            else:
                dict1[key2] = {key1: dict0[key1][key2]}
    return dict1


def list2str(list0, *args, sep='.'):
    """ Transform list0 into a string seperated by sep by the order of args.

        Parameters:
        --------
            list0: list
                List with form like: key1, value1, key2, value2...
            *args: list
                List of keys.
            sep: str
                Seperator betwwen two numbers in target string.

        Returns:
        --------
            res: str
                String seperated by sep.
    """
    res = ""
    if len(list0) == 0:
        res = "*"
    else:
        for arg in args:
            if arg in list0 or "-%s" % arg in list0:
                idx = list0.index(arg)
                if len(res) == 0:
                    res = "%s" % (list0[idx + 1])
                else:
                    res += "%s%s" % (sep, list0[idx + 1])
            else:
                if len(res) == 0:
                    res = "*"
                else:
                    res += "%s*" % (sep)
    return res


def get_para(file):
    """ Get parameters from filename.

        Parameters:
        --------
            file: str
                Name of input file.

        Returns:
        --------
            para: list
                eta, eps, Lx, Ly, seed
    """
    strList = (file.split("_")[1]).split(".")
    para = [int(i) for i in strList[:-1]]
    return para


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
                 outfile=None):
    """ Plot time serials of peak number and phi."""
    import matplotlib.pyplot as plt
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
                  (para[0] / 1000, para[1] / 1000, para[2], para[3], para[4]))
    plt.tight_layout()
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
    plt.close()


def plot_rhox_mean(para, num_set, sum_rhox, count_rhox, xlim=None, ylim=None):
    """ Plot time-averaged density profile rho_x.

        Parameters:
        --------
            para: list
                Parameters: eta, eps, Lx, Ly, seed
            num_set: np.ndarray
                Set of number of peaks.
            sum_rhox: np.ndarray
                Sum of rhox over time for each peak number, respectively.
            count_rhox: np.ndarray
                Count of rhox with the same peak number.
            xlim: tuple
                (xmin, xmax)
            ylim: tuple
                (ylim, ymax)
    """
    import matplotlib.pyplot as plt
    eta, eps, Lx, Ly, seed = para
    x = np.arange(Lx) + 0.5
    for i in range(num_set.size):
        rhox = sum_rhox[i] / count_rhox[i]
        plt.plot(x, rhox, label=r"$n_b=%d$" % num_set[i])
    plt.title(r"$\eta=%g,\ \epsilon=%g,\ L_x=%d,\ L_y=%d,\, \rm{seed}=%d$" %
              (eta / 1000, eps / 1000, Lx, Ly, seed))
    plt.legend(loc="best")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
    plt.close()


def handle(eta, eps, Lx, Ly, seed, t_beg=10000, h=1.8, show=False, out=False):
    """ Handle the data for given parameters.

        Parameters:
        --------
            eta: int
            eps: int
            Lx: int
            Ly: int
            Seed: int
            t_beg: int
                Time step that the system become equilibrium.
            h: float
            show: bool
                Whether to show the plot of results.
            out: bool
                Whether to output results.
    """

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
        seg_num, seg_idx0, seg_idx1, interp="cubic")
    para = [eta / 1000, eps / 1000, Lx, Ly, seed]
    if show:
        plot_serials(para, t_beg, t_end, peak.num_raw, peak.num_smoothed,
                     seg_num, seg_idx0, seg_idx1, seg_phi, beg_movAve,
                     end_movAve, phi_movAve)
        plot_rhox_mean(para, num_set, sum_rhox, count_rhox)
    if out:
        outfile = output()
        return outfile


def handle_files(para, t_beg=10000, out=True, show=False):
    """ Handle all matched files. """
    pat = list2str(para, "eta", "eps", "Lx", "Ly", "seed")
    print(pat)
    files = glob.glob("rhox_%s.bin" % pat)
    for i, file in enumerate(files):
        eta, eps, Lx, Ly, seed = get_para(file)
        try:
            outfile = handle(
                eta, eps, Lx, Ly, seed, out=out, t_beg=t_beg, show=show)
            print("%d of %d: %s" % (i + 1, len(files), outfile))
        except:
            print("Error for %s" % file)


def show_snaps(dt, para):
    """ Show snapshot of rho_x every dt frame. """
    import matplotlib.pyplot as plt
    pat = list2str(para, "eta", "eps", "Lx", "Ly", "seed")
    files = glob.glob("rhox_%s.bin" % pat)
    for file in files:
        print(file)
        eta, eps, Lx, Ly, seed = get_para(file)
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
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot")
    print(os.getcwd())
    try:
        if sys.argv[1] == "handle":
            handle_files(sys.argv[2:], t_beg=5000)
        elif sys.argv[1] == "snap":
            print(sys.argv[3:])
            show_snaps(int(sys.argv[2]), sys.argv[3:])
    except:
        print("Wrong args! Should be")
        print("(1) handle eta 350 eps 20 ...")
        print("(2) snap dt eta 350 eps 20 ...")
