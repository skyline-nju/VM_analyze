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


// void renormalize_2d_uint16(unsigned short *in, int ny_in, int nx_in,
//                            int *out, int ny_out, int nx_out);

template <typename T1, typename T2>
void renormalize_2d(T1 *in, int ny_in, int nx_in, T2 *out, int ny_out, int nx_out) {
  int bx = nx_in / nx_out;
  int by = ny_in / ny_out;
  for (int j_out = 0; j_out < ny_out; j_out++) {
    int y_beg = j_out * by;
    int y_end = y_beg + by;
    for (int i_out = 0; i_out < nx_out; i_out++) {
      int x_beg = i_out * bx;
      int x_end = x_beg + bx;
      int k_out = i_out + nx_out * j_out;
      for (int j_in = y_beg; j_in < y_end; j_in++) {
        for (int i_in = x_beg; i_in < x_end; i_in++) {
          out[k_out] += in[i_in + nx_in * j_in];
        }
      }
    }
  }
}