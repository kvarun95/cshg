import numpy as np
import cfdct3d

option = 'P'
ac = 0
nbscales = 3
nbdstz_coarse = 3
n_W = 3
W = np.zeros((n_W), dtype=np.int32)
W = np.array([1, 54, 1], dtype=np.int32)

n_nxs_io = 56
n_nys_io = 56
n_nzs_io = 56
nxs_io = np.zeros((56,), dtype=np.int32)
nys_io = np.zeros((56,), dtype=np.int32)
nzs_io = np.zeros((56,), dtype=np.int32)

N1 = 256
N2 = 256
N3 = 10

n_xre_io = N1*N2*N3
xre_io = np.zeros((n_xre_io,))
n_xim_io = N1*N2*N3
xim_io = np.zeros((n_xim_io,))

n_cre_io = 2
cre_io = np.zeros((n_cre_io,))
n_cim_io = 2
cim_io = np.zeros((n_cim_io,))

cfdct3d.call_fdct3d(xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W, N1, N2, N3, nbscales, nbdstz_coarse, ac, option)