/**
 * @brief Cal number fluctuation
 * 
 * @file num_flct.h
 * @author skyline-nju
 * @date 2018-09-04
 */

#pragma once

void cal_num_flct(unsigned short *num, int nz, int ny, int nx,
                  int *box_len, int *box_num, double *num_mean, double *num_var, int box_len_dim);