import numpy as np 
import pycfdct3d

def get_curvelet_params(input_shape, nbscales, nbdstz_coarse, ac=0):

    """ Parameters for computing the 3d curvelet transform. For more details, refer to Curvelab documentation.
    """

    assert(len(input_shape)==3), "Input shape must have length 3 for a 3d signal"
    N1,N2,N3 = input_shape

    lamda = 0.
    n_W = nbscales
    W = np.zeros((n_W,), dtype=np.int32)

    ns = 2
    ns_placeholder = np.zeros((ns,), dtype=np.int32)
    
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
                        ac, lamda, 'W')

    ns_io = sum(W)
    nxs_io = np.zeros((ns_io,), dtype=np.int32)
    nys_io = np.zeros((ns_io,), dtype=np.int32)
    nzs_io = np.zeros((ns_io,), dtype=np.int32)

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                        cre_io, cim_io,
                        nxs_io, nys_io, nzs_io, W,
                        N1, N2, N3, nbscales, nbdstz_coarse,
                        ac, lamda, 'P')

    nc = sum(nxs_io*nys_io*nzs_io)

    params = {'nxs': nxs_io,
              'nys': nys_io,
              'nzs': nzs_io,
              'W': W,
              'nbscales': nbscales,
              'nbdstz_coarse': nbdstz_coarse,
              'ac': ac,
              'input_shape': input_shape}

    return params


def curvedec3(x, params, mode="flat"):
    """ Computes the 3d forward discrete curvelet transform. 
    `x`     : Input complex 3d numpy ndarray
    `params`: Curvelet parameters obtained from `get_curvelet_params`
    `mode`  : If `"flat"`, return the coeffs in a flattened 1d array.
              If `"cell"`, return the coeffs in standard cell format.
    """

    assert x.shape==params['input_shape'], "Dimensions of input does not match the ones recorded in `params`."

    N1, N2, N3 = params['input_shape']
    nxs_io = params['nxs']
    nys_io = params['nys']
    nzs_io = params['nzs']
    W = params['W']
    nbscales = params['nbscales']
    nbdstz_coarse = params['nbdstz_coarse']
    ac = params['ac']
    option = 'F'
    lamda = 0.

    ns_io = sum(W)
    nc = sum(nxs_io*nys_io*nzs_io)

    cre_io = np.zeros((nc,))
    cim_io = np.zeros((nc,))

    xre_io = x.real.flatten().astype('double')
    xim_io = x.imag.flatten().astype('double')

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                    cre_io, cim_io,
                    nxs_io, nys_io, nzs_io, W,
                    N1, N2, N3, nbscales, nbdstz_coarse,
                    ac, lamda, option)

    c = cre_io + 1.j*cim_io

    if mode=="cell":
        raise NotImplementedError("Exporting curvelet coeffs in a cell package is not implemented.")
    
    return c


def curverec3(c, params, mode="flat"):
    """ Computes the inverse 3d curvelet transform.
    `c`      : Input complex curvelet coefficients (currently implemented only in the flattened form)
    `params` : Curvelet parameters obtained from `get_curvelet_params`
    `mode`   : If `"flat"`, `c` must be a 1d array.
               If `"cell"`, `c` must be in the Curvelab cell form. (Currently not implemented)
    """

    N1, N2, N3 = params['input_shape']
    nxs_io = params['nxs']
    nys_io = params['nys']
    nzs_io = params['nzs']
    W = params['W']
    nbscales = params['nbscales']
    nbdstz_coarse = params['nbdstz_coarse']
    ac = params['ac']
    option = 'B'
    lamda = 0.

    ns_io = sum(W)
    nc = sum(nxs_io*nys_io*nzs_io)

    if mode=="cell":
        raise NotImplementedError("Importing curvelet coeffs in a cell package is not implemented.")
    
    assert len(c)==nc, "Improper shape for curvelet coefficients."
    cre_io = c.real.astype('double')
    cim_io = c.imag.astype('double')

    xre_io = np.zeros((N1*N2*N3,)).astype('double')
    xim_io = np.zeros((N1*N2*N3,)).astype('double')

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                        cre_io, cim_io,
                        nxs_io, nys_io, nzs_io, W,
                        N1, N2, N3, nbscales, nbdstz_coarse,
                        ac, lamda, option)

    x = xre_io.reshape((N1,N2,N3)) + 1.j*xim_io.reshape((N1,N2,N3))
    return x


def curvesoft3(x, lamda, params):
    """ Soft thresholding in the curvelet domain implemented directly in c++.
    `x`     : Complex 3d numpy ndarray to be soft thresholded.
    `lamda` : Regularization parameter.
    `params`: Curvelet parameters obtained from `get_curelet_params()`
    """

    assert x.shape==params['input_shape'], "Dimensions of input does not match the ones recorded in `params`."

    N1, N2, N3 = params['input_shape']
    nxs_io = params['nxs']
    nys_io = params['nys']
    nzs_io = params['nzs']
    W = params['W']
    nbscales = params['nbscales']
    nbdstz_coarse = params['nbdstz_coarse']
    ac = params['ac']
    option = 'S'

    ns_io = sum(W)
    nc = sum(nxs_io*nys_io*nzs_io)

    cre_io = np.zeros((nc,))
    cim_io = np.zeros((nc,))

    xre_io = x.real.flatten().astype('double')
    xim_io = x.imag.flatten().astype('double')

    xre_io, xim_io, cre_io, cim_io, nxs_io, nys_io, nzs_io, W = pycfdct3d.pycall_fdct3d(xre_io, xim_io,
                    cre_io, cim_io,
                    nxs_io, nys_io, nzs_io, W,
                    N1, N2, N3, nbscales, nbdstz_coarse,
                    ac, lamda, option)

    x_th = xre_io.reshape(x.shape) + 1.j*xim_io.reshape(x.shape)
    return x_th

    
