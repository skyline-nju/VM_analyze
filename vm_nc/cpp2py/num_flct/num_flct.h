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

template <typename T1, typename T2>
void renormalize_3d(T1 *in, int nz_in, int ny_in, int nx_in,
                    T2 *out, int nz_out, int ny_out, int nx_out) {
  int bx = nx_in / nx_out;
  int by = ny_in / ny_out;
  int bz = nz_in / nz_out;
  for (int z_out = 0; z_out < nz_out; z_out++) {
    int z_beg = z_out * bz;
    int z_end = z_beg + bz;
    for (int y_out = 0; y_out < ny_out; y_out++) {
      int y_beg = y_out * by;
      int y_end = y_beg + by;
      for (int x_out = 0; x_out < nx_out; x_out++) {
        int x_beg = x_out * bx;
        int x_end = x_beg + bx;
        int k_out = x_out + nx_out * y_out + ny_out * nx_out * z_out;
        for (int z_in = z_beg; z_in < z_end; z_in++) {
          int nx_ny_z = nx_in * ny_in * z_in;
          for (int y_in = y_beg; y_in < y_end; y_in++) {
            int nx_y = nx_in * y_in;
            for (int x_in = x_beg; x_in < x_end; x_in++) {
              out[k_out] += in[x_in + nx_y + nx_ny_z];
            }
          }
        }
      }
    }
  }
}