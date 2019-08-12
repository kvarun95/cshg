
#include "fdct3d.hpp"
#include "fdct3dinline.hpp"


void call_fdct3d(double* xre_io, int n_xre_io,
                double* xim_io, int n_xim_io,
                double* cre_io, int n_cre_io,
                double* cim_io, int n_cim_io,
                int* nxs_io, int n_nxs_io,
                int* nys_io, int n_nys_io,
                int* nzs_io, int n_nzs_io,
                int* W, int n_W,
                int N1, int N2, int N3, 
                int nbscales, int nbdstz_coarse, 
                int ac, double lamda, char option);