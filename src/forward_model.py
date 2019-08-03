import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la
import scipy.sparse.linalg as sla

import utils
from mask import * 
from physical_params import *

""" Physically realistic forward model class and methods for optical microscopy.
"""

class fresnelprop(sla.LinearOperator):
    """ Fresnel propagator using the angular spectrum method.
    """

    def __init__(self, input_shape, output_shape, dtype='complex'):

        super(fresnelprop, self).__init__(
            dtype=dtype,
            shape=(np.prod(output_shape), np.prod(input_shape)),
            )

        self.shapeOI = [output_shape, input_shape]
        

    def _matvec(self, x):

        pass







