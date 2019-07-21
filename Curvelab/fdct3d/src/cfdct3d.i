%module cfdct3d
%{
    #define SWIG_FILE_WITH_INIT
    #include "cfdct3d.hpp"
    #include <iostream>
    #include "fftw.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* xre_io, int n_xre_io)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* xim_io, int n_xim_io)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* cre_io, int n_cre_io)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* cim_io, int n_cim_io)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* nxs_io, int n_nxs_io)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* nys_io, int n_nys_io)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* nzs_io, int n_nzs_io)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* W, int n_W)};

%include "cfdct3d.hpp"