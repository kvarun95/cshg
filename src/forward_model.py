import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la
import scipy.sparse.linalg as sla

import utils
from mask import * 
from physical_params import *

from numpy import pi, cos, sin, exp, sqrt

""" Physically realistic forward model class and methods for optical microscopy.
"""

class fresnelprop(sla.LinearOperator):
    """ Fresnel propagator using the angular spectrum method.
    """

    def __init__(self,
                input_shape,
                output_shape,
                dtype='complex',
                include_shg=False):

        super(fresnelprop, self).__init__(
            dtype=dtype,
            shape=(np.prod(output_shape), np.prod(input_shape)),
            )
        self.shapeOI = [output_shape, input_shape]

        assert input_shape[0]==input_shape[1], "Input X and Y shapes must be equal."        
        self.N = input_shape[0]
        self.NZ = input_shape[2]

        # get the spatial grid associated with the given input shape and length scales.
        spatial_grid = create_grid(self.N, self.NZ)
        dL = spatial_grid['dL']
        dF = spatial_grid['dF']
        dLZ = spatial_grid['dLZ']
        FMAX = spatial_grid['FMAX']
        Z = spatial_grid['Z']
        X = spatial_grid['X']
        gridx = spatial_grid['gridx']
        F = spatial_grid['F']
        gridf = spatial_grid['gridf']
        self.spatial_grid = spatial_grid

        magf2 = gridf[...,0]**2 + gridf[...,1]**2
        self.NAmask = float(magf2 >= (NA/WAVELENGTH)**2)
        self.kernel = np.zeros(input_shape, dtype=complex)

        for i in range(len(Z)):
            phase = -2.*pi*Z[i]*sqrt((1./WAVELENGTH)**2 - magf2)
            self.kernel[...,:] = utils.cis(phase) * self.NAmask

        self.adj_kernel = self.kernel.conj()

        if include_shg:
            raise NotImplementedError("Forward model simulation including the SHG has not been implemented yet")


    def _matvec(self, x):

        assert self.shapeOI[1]==x.shape, "Improper input or kernel shape"

        xf = np.fft.fft2(x, axes=(0,1))
        xf = np.fft.fftshift(xf, axes=(0,1))
        yf = self.kernel * xf 

        yf = np.fft.ifftshift(yf, axes=(0,1))
        y = np.fft.ifft2(yf, axes=(0,1))

        return np.sum(y, axis=2)


    def _adjoint(self, y):
        
        assert self.shapeOI[0]==y.shape, "Improper input or kernel shape"
        
        z = np.stack([y]*self.NZ, axis=2)
        zf = np.fft.fft2(z, axes=(0,1))
        zf = np.fft.fftshift(zf, axes=(0,1))

        xf = self.adj_kernel * zf
        xf = np.fft.ifftshift(xf, axes=(0,1))
        x = np.fft.ifft2(xf, axes=(0,1))

        return x        





