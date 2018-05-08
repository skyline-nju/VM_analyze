"""
Load correlation functions and snapshots from netcdf4 files.

@author skyline-nju
@date 2018-05-08
"""
from netCDF4 import Dataset
import numpy as np


class Corr2d:
    def __init__(self, file):
        self.rootgrp = Dataset(file, "r", format="NETCDF4")

    def gene_frame(self, beg=0, end=None, sep=1):
        nframes = len(self.rootgrp.dimensions["frame"])
        if end is None:
            end = nframes
        for i in range(beg, end, sep):
            t = self.rootgrp.variables["time"][i: i+1][0]
            vm = self.rootgrp.variables["mean_velocity"][i: i+1][0]
            c_rho = np.array(self.rootgrp.variables["C_rho"][i: i+1][0])
            c_v = np.array(self.rootgrp.variables["C_v"][i: i+1][0])
            yield [t, vm, c_rho, c_v]


class CGSnap:
    def __init__(self, file):
        self.rootgrp = Dataset(file, "r", format="NETCDF4")

    def gene_frame(self, beg=0, end=None, sep=1):
        nframes = len(self.rootgrp.dimensions["frame"])
        if end is None:
            end = nframes
        for i in range(beg, end, sep):
            t = self.rootgrp.variables["time"][i: i+1][0]
            num = self.rootgrp.variables["num"][i: i+1][0]
            vx = self.rootgrp.variables["vx"][i: i+1][0]
            vy = self.rootgrp.variables["vy"][i: i+1][0]
            yield [t, num, vx, vy]


if __name__ == "__main__":
    import os
    # import matplotlib.pyplot as plt
    os.chdir(r"D:\tmp")
    f = Corr2d(r"cr_0.18_0_1_512_512_1.nc")
    f = CGSnap(r"cg_0.18_0_1_512_512_1.nc")
    frames = f.gene_frame()
    for frame in frames:
        t, vm, c_rho, c_v = frame
        # t, num, vx, vy = frame
        # print(t)
        # print(c_v[0, 0])
        # plt.imshow(c_rho, origin="lower")
        # plt.show()
        # plt.close()
