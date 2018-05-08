/**
 * @brief calculate 2d spatial correlation function for coarse-grained density
 * and velocity field.
 * 
 * @file spatial_corr.h
 * @author skyline-nju
 * @date 2018-05-07
 */
#include "spatial_corr.h"
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;
/*************************************************************************//**
 * @brief Construct a new Auto Corr 2d:: Auto Corr 2d object
 * 
 * @param _nrow   num of cols
 * @param _ncol   num of rows
 ****************************************************************************/
AutoCorr2d::AutoCorr2d(int _nrow, int _ncol) {
  // shape of input/output array = (_nrows, _ncols)
  const auto ncol_complex = _ncol / 2 + 1;
  ntot_ = _nrow * _ncol;
  ntot_complex_ = _nrow * ncol_complex;
  in_ = (double *)fftw_malloc(sizeof(double) * ntot_);
  inter_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ntot_complex_);
  out_ = (double *)fftw_malloc(sizeof(double) * ntot_);
  fft_ = fftw_plan_dft_r2c_2d(_nrow, _ncol, in_, inter_, FFTW_MEASURE);
  rfft_ = fftw_plan_dft_c2r_2d(_nrow, _ncol, inter_, out_, FFTW_MEASURE);
  fftw_free(in_);
  fftw_free(out_);
}

AutoCorr2d::~AutoCorr2d() {
  fftw_destroy_plan(fft_);
  fftw_destroy_plan(rfft_);
  fftw_free(inter_);
}

void AutoCorr2d::autocorr2(double * in_new, double * out_new) const {
  fftw_execute_dft_r2c(fft_, in_new, inter_);
  for (size_t i = 0; i < ntot_complex_; i++) {
    inter_[i][0] = inter_[i][0] * inter_[i][0] + inter_[i][1] * inter_[i][1];
    inter_[i][1] = 0.0;
  }
  fftw_execute_dft_c2r(rfft_, inter_, out_new);
  for (size_t i = 0; i < ntot_; i++) {
    out_new[i] /= ntot_;
  }
}

void AutoCorr2d::autocorr2(double * in_new, double * out_new, double *Sk) const {
  fftw_execute_dft_r2c(fft_, in_new, inter_);
  for (size_t i = 0; i < ntot_complex_; i++) {
    inter_[i][0] = inter_[i][0] * inter_[i][0] + inter_[i][1] * inter_[i][1];
    inter_[i][1] = 0.0;
    Sk[i] = inter_[i][0];
  }
  fftw_execute_dft_c2r(rfft_, inter_, out_new);
  for (size_t i = 0; i < ntot_; i++) {
    out_new[i] /= ntot_;
  }
}

/*************************************************************************//**
 * @brief Construct a new Spatial Corr 2d:: Spatial Corr 2d object
 * 
 * @param ncols0     number of cols
 * @param nrows0     number of rows
 * @param Lx         Domain length in x
 * @param Ly         Domain length in y
 ****************************************************************************/
SpatialCorr2d::SpatialCorr2d(int ncols0, int nrows0, double Lx, double Ly)
  : AutoCorr2d(nrows0, ncols0), ncols_(ncols0), nrows_(nrows0) {
  ncells_ = ncols_ * nrows_;
  lx_ = Lx / ncols_;
  ly_ = Ly / nrows_;
  cell_area_ = lx_ * ly_;
}

CircleAverage::CircleAverage(int _ncols, int _nrows, double l, vector<double> &r_arr) :
  ncols(_ncols), nrows(_nrows) {
  half_nrows = nrows / 2;
  half_ncols = ncols / 2;
  RR_max = half_nrows * half_nrows;
  std::map<unsigned int, unsigned int> dict_count;
  for (int row = 0; row < half_nrows; row++) {
    unsigned int yy = row * row;
    if (row == 0) {
      for (int col = 0; col < half_ncols; col++) {
        unsigned int  RR = col * col + yy;
        if (RR < RR_max) {
          dict_count[RR] += 1;
        }
      }
    } else {
      for (int col = 0; col < ncols; col++) {
        int dx = col < half_ncols ? col : ncols - col;
        unsigned int RR = dx * dx + yy;
        if (RR < RR_max) {
          dict_count[RR] += 1;
        }
      }
    }
  }

  R_thresh = 10;
  RR_thresh = R_thresh * R_thresh;
  int r_ceiling = R_thresh;
  int rr_ceiling = RR_thresh;
  auto it = dict_count.cbegin();
  while (it != dict_count.cend()) {
    if (it->first < RR_thresh) {
      r_arr.push_back(sqrt(it->first));
      count.push_back(it->second);
      ++it;
    } else {
      if (rr_ceiling == RR_thresh)
        idx_thresh = r_arr.size();
      rr_ceiling += 2 * r_ceiling + 1;
      r_ceiling++;
      double sum_r = 0;
      int sum_count = 0;
      while (it->first < rr_ceiling && it != dict_count.cend()) {
        sum_r += sqrt(it->first) * it->second;
        sum_count += it->second;
        ++it;
      }
      r_arr.push_back(sum_r / sum_count);
      count.push_back(sum_count);
    }
  }
  size_r = r_arr.size();
  for (int i = 0; i < size_r; i++) {
    r_arr[i] *= l;
  }
}

void CircleAverage::eval(const double *corr2d, double *corr_r) const {
  for (int i = 0; i < size_r; i++) {
    corr_r[i] = 0;
  }
  map<unsigned int, double> corr1d;
  for (int row = 0; row < half_nrows; row++) {
    if (row == 0) {
      for (int col = 0; col < half_ncols; col++) {
        unsigned int rr = col * col;
        if (rr < RR_thresh) {
          corr1d[rr] += corr2d[col];
        } else if (rr < RR_max) {
          int idx = idx_thresh + col - R_thresh;
          corr_r[idx] += corr2d[col];
        }
      }
    } else {
      unsigned int yy = row * row;
      for (int col = 0; col < ncols; col++) {
        int x = col < half_ncols ? col : ncols - col;
        unsigned int rr = yy + x * x;
        if (rr < RR_thresh) {
          corr1d[rr] += corr2d[col + ncols * row];
        } else if (rr < RR_max) {
          int idx = idx_thresh + int(sqrt(rr)) - R_thresh;
          corr_r[idx] += corr2d[col + ncols * row];
        }
      }
    }
  }
  int j = 0;
  for (auto it = corr1d.cbegin(); it != corr1d.cend(); ++it) {
    corr_r[j] = it->second / count[j];
    j++;
  }
  for (; j < size_r; j++) {
    corr_r[j] /= count[j];
  }
}