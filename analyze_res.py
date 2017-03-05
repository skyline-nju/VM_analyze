import numpy as np
import os
import multBand as mb

if __name__ == "__main__":
    os.chdir("E:\\data\\random_torque\\bands\\Lx\\snapshot\\rhox")
    file = "mb_350.0.740.200.214740.npz"
    buff = np.load(file)
    para = mb.get_para(file)
    mb.plot_rhox_mean(para, buff["num_set"], buff["sum_rhox"], buff["count_rhox"])

