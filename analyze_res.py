import numpy as np
import os
import glob
import multBand as mb


def plot_all_serials():
    files = glob.glob("mb_*.npz")
    for file in files:
        bf = np.load(file)
        para = mb.get_para(file)
        outfile = "ts_%d.%d.%d.%d.%d.png" % (para[0], para[1], para[2],
                                             para[3], para[4])
        t_beg, t_end = bf["t_beg_end"]
        mb.plot_serials(
            para,
            t_beg,
            t_end,
            bf["num_raw"],
            bf["num_smoothed"],
            bf["seg_num"],
            bf["seg_idx0"],
            bf["seg_idx1"],
            bf["seg_phi"],
            bf["beg_movAve"],
            bf["end_movAve"],
            bf["phi_movAve"],
            outfile=outfile)


if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    # file = "mb_350.0.740.200.214740.npz"
    # buff = np.load(file)
    # para = mb.get_para(file)
    # mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"],
    #                   buff["count_rhox"])
    plot_all_serials()