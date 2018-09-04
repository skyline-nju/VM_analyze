/**
 * @brief Cal number fluctuation
 * 
 * @file num_flct.cpp
 * @author skyline-nju
 * @date 2018-09-04
 */
#include "num_flct.h"

int get_sum(const unsigned short *mat, int ny, int nx, const int startset[3], const int countset[3]) {
  int res = 0;
  for (int z = startset[0]; z < startset[0] + countset[0]; z++) {
    const int z_ny_nx = z * ny * nx;
    for (int y = startset[1]; y < startset[1] + countset[1]; y++) {
      const int y_nx = y * nx;
      for (int x = startset[2]; x < startset[2] + countset[2]; x++) {
        const int i = x + y_nx + z_ny_nx;
        res += mat[i];
      }
    }
  }
  return res;
}

void cal_num_flct(unsigned short *num, int nz, int ny, int nx,
                  int *box_len, int *box_num, double *num_mean, double *num_var, int box_len_dim) {
  for (int i = 0; i < box_len_dim; i++) {
    int len = box_len[i];
    const int countset[3] = { len, len, len };
    double num_sum = 0;
    double num2_sum = 0;
    for (int iz = 0; iz < nz; iz += len) {
      for (int iy = 0; iy < ny; iy += len) {
        for (int ix = 0; ix < nx; ix += len) {
          const int startset[3] = { iz, iy, ix };
          const double n_block = get_sum(num, ny, nx, startset, countset);
          num_sum += n_block;
          num2_sum += n_block * n_block;
        }
      }
    }
    num_mean[i] = num_sum / box_num[i];
    num_var[i] = num2_sum / box_num[i] - num_mean[i] * num_mean[i];
  }
}