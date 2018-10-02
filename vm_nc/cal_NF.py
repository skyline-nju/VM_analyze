"""
Cal number fluctuations in 3D domain.

2018/9/4
"""
import os
import numpy as np
from read_nc import get_host_info, read_densities_3


def cal_nf_3(densities, box_len, box_num):
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


def time_ave_num_flct_3(infile, first_frame):
    from cpp2py.num_flct import cal_num_flct_3
    host_size, cells_per_host = get_host_info(infile)
    nx, ny, nz = host_size * cells_per_host
    if nx == 60:
        box_len = np.array([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30], np.int32)
    elif nx == 120:
        box_len = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12,
                            15, 20, 24, 30, 40, 60], np.int32)
    box_num = np.array([nx * ny * nz // i ** 3 for i in box_len], np.int32)

    n_mean = np.zeros(box_len.size)
    n_var = np.zeros_like(n_mean)
    count = 0
    frames = read_densities_3(infile, first_frame)
    for frame in frames:
        t, densities = frame
        n_mean_t = np.zeros(box_len.size)
        n_var_t = np.zeros(box_len.size)
        cal_num_flct_3(densities, box_len, box_num, n_mean_t, n_var_t)
        n_mean += n_mean_t
        n_var += n_var_t
        count += 1
    n_mean /= count
    n_var /= count
    return n_mean, n_var


if __name__ == "__main__":
    os.chdir(r"D:\data\vm3d")
    infile = "field_240_0.20_0.060_1.0_12.nc"
    n_mean, n_std = time_ave_num_flct_3(infile, 20)
    for i in range(n_mean.size):
        print(n_mean[i], n_std[i])

    # os.chdir(r"D:\data\random_torque\large_system")
    # file0 = "field_480_0.20_0.000_1.0_1_host0.nc"
    # file1 = "field_480_0.20_0.000_1.0_1_host1.nc"
    # rootgrp0 = Dataset(file0, "r", format="NETCDF4")
    # rootgrp1 = Dataset(file1, "r", format="NETCDF4")
    # host_size = rootgrp0.variables["host_size"][:]
    # host_ny = rootgrp0.dimensions["NY"].size
    # host_nx = rootgrp0.dimensions["NX"].size
    # gl_ny = host_ny * host_size[0]
    # gl_nx = host_nx * host_size[1]
    # nframe = rootgrp0.dimensions["frame"].size

    # for i in range(nframe):
    #     print("t =", rootgrp0.variables["time"][i])
    #     den0 = rootgrp0.variables["density_field"][i, :, :]
    #     den1 = rootgrp1.variables["density_field"][i, :, :]
    #     # print(np.sum(den0) + np.sum(den1))
    # frames = read_densities_2(file0)
    # for frame in frames:
    #     den = frame
    #     print(np.mean(den))
