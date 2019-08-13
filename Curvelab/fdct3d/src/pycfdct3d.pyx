# distutils: language=c++

cdef extern from "cfdct3d.hpp":
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
                    int ac, double lamda, char option)

import numpy as np

def pycall_fdct3d(xre, xim, cre, cim, nxs, nys, nzs, w, N1, N2, N3, nbscales, nbdstz_coarse, ac, lamda, option):

    xre = xre.astype('double')
    xim = xim.astype('double')
    cre = cre.astype('double')
    cim = cim.astype('double')
    nxs = nxs.astype(np.int32)
    nys = nys.astype(np.int32)
    nzs = nzs.astype(np.int32)
    w = w.astype(np.int32)
    N1 = np.int32(N1)
    N2 = np.int32(N2)
    N3 = np.int32(N3)
    nbscales = np.int32(nbscales)
    nbdstz_coarse = np.int32(nbdstz_coarse)
    ac = np.int32(ac)

    if not xre.flags['C_CONTIGUOUS']:
        xre = np.ascontiguousarray(xre) # Makes a contiguous copy of the numpy array.
    if not xim.flags['C_CONTIGUOUS']:
        xim = np.ascontiguousarray(xim)
    if not cre.flags['C_CONTIGUOUS']:
        cre = np.ascontiguousarray(cre)
    if not cim.flags['C_CONTIGUOUS']:
        cim = np.ascontiguousarray(cim)
    if not nxs.flags['C_CONTIGUOUS']:
        nxs = np.ascontiguousarray(nxs)
    if not nys.flags['C_CONTIGUOUS']:
        nys = np.ascontiguousarray(nys)
    if not nzs.flags['C_CONTIGUOUS']:
        nzs = np.ascontiguousarray(nzs)
    if not w.flags['C_CONTIGUOUS']:
        w = np.ascontiguousarray(w)

    cdef double[::1] xre_mv = xre 
    cdef double[::1] xim_mv = xim 
    cdef double[::1] cre_mv = cre
    cdef double[::1] cim_mv = cim
    cdef int[::1] nxs_mv = nxs
    cdef int[::1] nys_mv = nys
    cdef int[::1] nzs_mv = nzs 
    cdef int[::1] w_mv = w 
    
    call_fdct3d(&xre_mv[0], xre_mv.shape[0],
                &xim_mv[0], xim_mv.shape[0],
                &cre_mv[0], cre_mv.shape[0],
                &cim_mv[0], cim_mv.shape[0],
                &nxs_mv[0], nxs_mv.shape[0],
                &nys_mv[0], nys_mv.shape[0],
                &nzs_mv[0], nzs_mv.shape[0],
                &w_mv[0], w_mv.shape[0],
                N1, N2, N3, nbscales, nbdstz_coarse, ac, lamda, ord(option))

    return xre, xim, cre, cim, nxs, nys, nzs, w