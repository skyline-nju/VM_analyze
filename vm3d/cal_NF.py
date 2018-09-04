"""
Cal number fluctuations.

2018/9/4
"""
import os
import numpy as np
from netCDF4 import Dataset


def cal_num_fluctuation(densities, box_len, box_num):
    n_sum = np.zeros(box_len.size)
    n2_sum = np.zeros(box_len.size)
    for i in range(box_len.size):
        box_len = box_len[i]
        for iz in range(densities.shape[0] // box_len):
            iz_beg = iz * box_len
            iz_end = iz_beg + box_len
            for iy in range(densities.shape[1] // box_len):
                iy_beg = iy * box_len
                iy_end = iy_beg + box_len
                for ix in range(densities.shape[2] // box_len):
                    ix_beg = ix * box_len
                    ix_end = ix_beg + box_len
                    n_block = np.sum(densities[iz_beg:iz_end,
                                               iy_beg:iy_end, ix_beg:ix_end])
                    n_sum[i] += n_block
                    n2_sum[i] += n_block ** 2
    n_mean = n_sum / box_num
    n_var = np.array([n2_sum[i] / box_num[i] - n_mean[i]
                      ** 2 for i in range(n_mean.size)])
    return n_mean, n_var


def time_ave_num_flct(infile, first_frame):
    from cpp2py.num_flct import cal_num_flct
    rootgrp = Dataset(infile, "r", format="NETCDF4")
    nx = rootgrp.dimensions["global_field_x"].size
    ny = rootgrp.dimensions["global_field_y"].size
    nz = rootgrp.dimensions["global_field_z"].size
    if nx == 60:
        box_len = np.array([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30], np.int32)
    elif nx == 120:
        box_len = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12,
                            15, 20, 24, 30, 40, 60], np.int32)
    box_num = np.array([nx * ny * nz // i ** 3 for i in box_len], np.int32)

    n_mean = np.zeros(box_len.size)
    n_var = np.zeros_like(n_mean)
    count = 0
    nframe = rootgrp.dimensions["frame"].size
    print("total frame:", nframe)
    for i in range(first_frame, nframe):
        print("t =", rootgrp.variables["time"][i])
        densities = rootgrp.variables["density_field"][i, :, :, :]
        # n_mean_t, n_var_t = cal_num_fluctuation(densities, box_len, box_num)
        n_mean_t = np.zeros(box_len.size)
        n_var_t = np.zeros(box_len.size)
        cal_num_flct(densities, box_len, box_num, n_mean_t, n_var_t)
        n_mean += n_mean_t
        n_var += n_var_t
        count += 1
    n_mean /= count
    n_var /= count
    return n_mean, n_var


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d")
    infile = "field_120_0.20_0.000_1.0_12.nc"
    n_mean, n_std = time_ave_num_flct(infile, 20)
    for i in range(n_mean.size):
        print(n_mean[i], n_std[i])
