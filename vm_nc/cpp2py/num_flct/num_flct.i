%define DOCSTRING
"Cal number fluctuations."
%enddef

%module (docstring=DOCSTRING) num_flct
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /* Includes the header in the wrapper code */
    #include "num_flct.h"
%}

/* include the numpy typemaps */
%include "D:\code\numpy.i"

/* need this for correct module initialization */
%init %{
    import_array();
%}

%feature("autodoc", "1");

%apply (unsigned short* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
	(unsigned short *num, int nz, int ny, int nx),
	(unsigned short *in, int nz_in, int ny_in, int nx_in)}

%apply (int *INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
	(int *out, int nz_out, int ny_out, int nx_out)
}

%apply (double * IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
	(double *in, int nz_in, int ny_in, int nx_in)
}

%apply (double * INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
	(double *out, int nz_out, int ny_out, int nx_out)
}

%apply (int* IN_ARRAY1, int DIM1) {(int *box_len, int box_len_dim),
								   (int *box_num, int box_num_dim)}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *num_mean, int num_mean_dim),
                                      (double *num_var, int num_var_dim)}

%apply (unsigned short* IN_ARRAY2, int DIM1, int DIM2) {
    (unsigned short *in, int ny_in, int nx_in)}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
	(double* in, int ny_in, int nx_in)}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (int *out, int ny_out, int nx_out)}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
	(double* out, int ny_out, int nx_out)}

//%rename (cal_num_flct) my_cal_num_flct;

%inline %{
    void cal_num_flct_3(unsigned short *num, int nz, int ny, int nx,
                         int *box_len, int box_len_dim, int *box_num, int box_num_dim,
                         double *num_mean, int num_mean_dim, double *num_var, int num_var_dim) {
        cal_num_flct(num, nz, ny, nx, box_len, box_num, num_mean, num_var, box_len_dim);
    }

    void renormalize_2d_uint16(unsigned short *in, int ny_in, int nx_in,
                               int *out, int ny_out, int nx_out) {
        renormalize_2d(in, ny_in, nx_in, out, ny_out, nx_out);
    }

	void renormalize_2d_doub(double *in, int ny_in, int nx_in,
							 double *out, int ny_out, int nx_out) {
		renormalize_2d(in, ny_in, nx_in, out, ny_out, nx_out);
	}

	void renormalize_3d_uint16(unsigned short *in, int nz_in, int ny_in, int nx_in,
							   int *out, int nz_out, int ny_out, int nx_out) {
		renormalize_3d(in, nz_in, ny_in, nx_in, out, nz_out, ny_out, nx_out);
	}

	void renormalize_3d_doub(double *in, int nz_in, int ny_in, int nx_in,
							 double *out, int nz_out, int ny_out, int nx_out) {
		renormalize_3d(in, nz_in, ny_in, nx_in, out, nz_out, ny_out, nx_out);
	}
%}

%include "num_flct.h"