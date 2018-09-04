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
%include "C:\Users\duany\Documents\code\numpy.i"

/* need this for correct module initialization */
%init %{
    import_array();
%}

%feature("autodoc", "1");

%apply (unsigned short* IN_ARRAY3, int DIM1, int DIM2, int DIM3) 
        {(unsigned short *num, int nz, int ny, int nx)}

%apply (int* IN_ARRAY1, int DIM1) {(int *box_len, int box_len_dim),
								   (int *box_num, int box_num_dim)}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *num_mean, int num_mean_dim),
                                      (double *num_var, int num_var_dim)}

%rename (cal_num_flct) my_cal_num_flct;

%inline %{
    void my_cal_num_flct(unsigned short *num, int nz, int ny, int nx,
                         int *box_len, int box_len_dim, int *box_num, int box_num_dim,
                         double *num_mean, int num_mean_dim, double *num_var, int num_var_dim) {
        cal_num_flct(num, nz, ny, nx, box_len, box_num, num_mean, num_var, box_len_dim);
    }
%}

%include "num_flct.h"