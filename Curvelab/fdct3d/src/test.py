import numpy as np
import cfdct3d
import pycfdct3d



def test_w():
# if True:

    option = 'W'
    ac = 0
    nbscales = 2
    nbdstz_coarse = 3
    n_W = nbscales
    W = np.zeros((n_W,), dtype=np.int32)

    ns = 2
    ns_placeholder = np.zeros((ns,), dtype=np.int32)
    
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

    xre, xim, cre, cim, nxs, nys, nzs, w = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                                                                   cre_io, cim_io,
                                                                   ns_placeholder, ns_placeholder, ns_placeholder, W,
                                                                   N1, N2, N3, nbscales, nbdstz_coarse,
                                                                   ac, option)

    print(w)

def test_param():
# if True:

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

    xre, xim, cre, cim, nxs, nys, nzs, w = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                                                                   cre_io, cim_io,
                                                                   nxs_io, nys_io, nzs_io, W,
                                                                   N1, N2, N3, nbscales, nbdstz_coarse,
                                                                   ac, option)

    i=0
    for s in range(nbscales):
        for w in range(W[s]):
            print("Dimension of c[",s,"][",w,"] is : (",nxs[i],",",nys[i],",",nzs[i],")")
            i = i+1


def test_forward():
# if True:

    option = 'F'
    ac = 0
    nbscales = 2
    nbdstz_coarse = 3
    n_W = nbscales
    W = np.zeros((n_W,), dtype=np.int32)

    ns = 2
    ns_placeholder = np.zeros((ns,), dtype=np.int32)
    
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

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                        cre_io, cim_io,
                        ns_placeholder, ns_placeholder, ns_placeholder, W,
                        N1, N2, N3, nbscales, nbdstz_coarse,
                        ac, 'W')

    print("W :", W)

    ns_io = sum(W)
    nxs_io = np.zeros((ns_io,), dtype=np.int32)
    nys_io = np.zeros((ns_io,), dtype=np.int32)
    nzs_io = np.zeros((ns_io,), dtype=np.int32)

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                        cre_io, cim_io,
                        nxs_io, nys_io, nzs_io, W,
                        N1, N2, N3, nbscales, nbdstz_coarse,
                        ac, 'P')

    nc = sum(nxs_io*nys_io*nzs_io)
    cre_io = np.zeros((nc,))
    cim_io = np.zeros((nc,))

    xre_io = np.random.rand(N1*N2*N3).astype('double')
    xim_io = np.random.rand(N1*N2*N3).astype('double')

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                        cre_io, cim_io,
                        nxs_io, nys_io, nzs_io, W,
                        N1, N2, N3, nbscales, nbdstz_coarse,
                        ac, option)







