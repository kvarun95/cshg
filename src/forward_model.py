import numpy as np 
import scipy.linalg as la
import scipy.sparse.linalg as sla

import utils
from mask import * 
from physical_params import *

from numpy import pi, cos, sin, exp, sqrt

""" Physically realistic forward model class and methods for optical microscopy.
"""

class FresnelProp(object):
    """ Fresnel propagator using the angular spectrum method.
    """

    def __init__(self,
                input_shape,
                output_shape,
                dtype=complex,
                defocus=0.,
                include_shg=False):

        self.shapeOI = [output_shape, input_shape]
        self.shape = ( np.prod(output_shape), np.prod(input_shape) )
        self.dtype = dtype
        self.defocus = defocus
        self.include_shg = include_shg

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
        self.NAmask = (magf2 <= (NA/WAVELENGTH)**2).astype(float)
        self.kernel = np.zeros(input_shape, dtype=complex)

        d0 = defocus

        if include_shg:
            for i in range(len(Z)):
                # starting phase due to propagation of fundamental, and phase matching
                phase0 = -2.*pi* (REF_IDX/WAVELENGTH) *Z[i]
                phase = -2.*pi* (LZ-Z[i]-d0) *sqrt((REF_IDX/WAVELENGTH)**2 - magf2)
                self.kernel[...,i] = utils.cis(phase) * self.NAmask * utils.cis(phase0)
    
            self.adj_kernel = self.kernel.conj()

        else:
            for i in range(len(Z)):
                phase = -2.*pi* (LZ-Z[i]-d0) *sqrt((REF_IDX/WAVELENGTH)**2 - magf2)
                self.kernel[...,i] = utils.cis(phase) * self.NAmask
    
            self.adj_kernel = self.kernel.conj()


    def __call__(self, x):
        return self._matvec(x)


    def adjoint(self, x):
        return self._adjoint(x)


    def _matvec(self, x):

        # assert self.shapeOI[1]==x.shape, "Improper input or kernel shape"

        xf = np.fft.fft2(x, axes=(0,1))
        xf = np.fft.fftshift(xf, axes=(0,1))
        yf = self.kernel * xf 

        yf = np.fft.ifftshift(yf, axes=(0,1))
        y = np.fft.ifft2(yf, axes=(0,1))

        return np.sum(y, axis=2)


    def _adjoint(self, y):
        
        # assert self.shapeOI[0]==y.shape, "Improper input or kernel shape"
        
        z = np.stack([y]*self.NZ, axis=2)
        zf = np.fft.fft2(z, axes=(0,1))
        zf = np.fft.fftshift(zf, axes=(0,1))

        xf = self.adj_kernel * zf
        xf = np.fft.ifftshift(xf, axes=(0,1))
        x = np.fft.ifft2(xf, axes=(0,1))

        return x


class FourierProp(object):
    """ Fourier optics lens propagation model. Described as:
    Fourier lens -> mask -> fourier lens 
    """
    
    def __init__(self, mask, dtype=complex):
        
        self.shape=( np.prod(mask.shape) * mask.num, np.prod(mask.shape) )
        self.shapeOI = ( (mask.num, *mask.shape), mask.shape )

        self.mask = mask
        self.dtype = dtype

    def __call__(self, x):
        return self._matvec(x)

    def adjoint(self, x):
        return self._adjoint(x)


    def _matvec(self, x):
        
        # assert self.shapeOI[1]==x.shape, "Improper input or kernel shape"

        xf = np.fft.fft2(x, axes=(-1,-2), norm='ortho')
        xf = np.fft.fftshift(xf, axes=(-1,-2))

        xfm = self.mask.apply(xf)

        yf = np.fft.ifftshift(xfm, axes=(-1,-2))
        y = np.fft.ifft2(yf, axes=(-1,-2), norm='ortho')

        return y


    def _adjoint(self, y):

        # assert self.shapeOI[0]==y.shape, "Improper input or kernel shape"

        yf = np.fft.fft2(y, axes=(-1,-2), norm='ortho')
        yf = np.fft.fftshift(yf, axes=(-1,-2))

        yfm = self.mask.apply(yf)
        xf = np.sum(yfm, axis=0)

        xf = np.fft.ifftshift(xf, axes=(-1,-2))
        x = np.fft.ifft2(xf, axes=(-1,-2), norm='ortho')

        return x



class ForwardModel(object):

    def __init__(self, input_shape, mask, defocus=0., include_shg=True):

        assert mask.shape==input_shape[:2], "Improper shapes for mask and input"

        self.dtype=complex,
        self.shape=( np.prod(mask.shape)*mask.num, np.prod(input_shape) )
        self.shapeOI = ( (mask.num, *mask.shape), input_shape )
        self.defocus = defocus
        self.include_shg = include_shg

        self.uscope = FresnelProp(input_shape, mask.shape, defocus=self.defocus, include_shg=self.include_shg)
        self.fourf = FourierProp(mask)


    def __call__(self, x):
        return self._matvec(x)

    def adjoint(self, y):
        return self._adjoint(y)
    
    def _matvec(self, x):
        return self.fourf(self.uscope(x))

    def _adjoint(self, y):
        return self.uscope.adjoint(self.fourf.adjoint(y))