/**
 * @brief calculate 2d spatial correlation function for coarse-grained density
 * and velocity field.
 * 
 * @file spatial_corr.h
 * @author skyline-nju
 * @date 2018-05-07
 */
#ifndef SPATIAL_CORR_H
#define SPATIAL_CORR_H
#include <vector>
#include <fftw3.h>

class AutoCorr2d {
public:
  AutoCorr2d(int _nrow, int _ncol);
  ~AutoCorr2d();
  void autocorr2(double * in_new, double * out_new) const;
  void autocorr2(double * in_new, double * out_new, double *Sk) const;
private:
  size_t ntot_;
  size_t ntot_complex_;

  double * in_;
  fftw_complex *inter_;
  double * out_;

  fftw_plan fft_;
  fftw_plan rfft_;
};

class SpatialCorr2d : public AutoCorr2d {
public:
  SpatialCorr2d(int ncols0, int nrows0, double Lx, double Ly);
  template <typename TPar,typename T>
  void cal_corr(const TPar *p_arr, int n_par, T *c_rho, T *c_v,
                double &vx_tot, double &vy_tot) const;

private:
  template <typename TPar>
  void coarse_grain(const TPar *p_arr, int n_par, double *rho,
                    double *vx, double *vy, double &vx_tot, double &vy_tot) const;
protected:
  int ncols_;
  int nrows_;
  size_t ncells_;
  double lx_;
  double ly_;
  double cell_area_;
};

template <typename TPar, typename T>
void SpatialCorr2d::cal_corr(const TPar* p_arr, int n_par, T* c_rho, T* c_v,
                             double& vx_tot, double& vy_tot) const {
  double *rho = (double *)fftw_malloc(sizeof(double) * ncells_);
  double *vx = (double *)fftw_malloc(sizeof(double) * ncells_);
  double *vy = (double *)fftw_malloc(sizeof(double) * ncells_);
  for (size_t i = 0; i < ncells_; i++) {
    rho[i] = vx[i] = vy[i] = 0;
  }
  coarse_grain(p_arr, n_par, rho, vx, vy, vx_tot, vy_tot);

  double *corr_rho = (double *)fftw_malloc(sizeof(double) * ncells_);
  double *corr_vx = (double *)fftw_malloc(sizeof(double) * ncells_);
  double *corr_vy = (double *)fftw_malloc(sizeof(double) * ncells_);

  autocorr2(rho, corr_rho);
  autocorr2(vx, corr_vx);
  autocorr2(vy, corr_vy);

  for (size_t i = 0; i < ncells_ / 2; i++) {
    c_rho[i] = corr_rho[i] / ncells_;
    c_v[i] = (corr_vx[i] + corr_vy[i]) / corr_rho[i];
  }

  fftw_free(rho);
  fftw_free(vx);
  fftw_free(vy);
  fftw_free(corr_rho);
  fftw_free(corr_vx);
  fftw_free(corr_vy);
}

template <typename TPar>
void SpatialCorr2d::coarse_grain(const TPar* p_arr, int n_par, double* rho,
                                 double* vx, double* vy,
                                 double &vx_tot, double &vy_tot) const {
  vx_tot = vy_tot = 0;
  for (auto i = 0; i < n_par; i++) {
    const auto col = int(p_arr[i].x / lx_);
    const auto row = int(p_arr[i].y / ly_);
    const auto i_cell = col + row * ncols_;
    rho[i_cell] += 1 / cell_area_;
    vx[i_cell] += p_arr[i].vx;
    vy[i_cell] += p_arr[i].vy;
    vx_tot += p_arr[i].vx;
    vy_tot += p_arr[i].vy;
  }
}

class CircleAverage {
public:
  CircleAverage(int _ncols, int _nrows, double l, std::vector<double> &r);
  void eval(const double *corr2d, double *corr_r) const;
private:
  int ncols;
  int nrows;
  int half_nrows;
  int half_ncols;
  unsigned int RR_max;
  int R_thresh;
  unsigned int RR_thresh;
  int idx_thresh;
  int size_r;
  std::vector<unsigned int> count;
};

#endif
