import numpy as np 
import scipy.linalg as la 
import scipy.sparse.linalg as sla

import utils
from forward_model import *
from physical_params import *
from mask import *

from numpy import cos, sine, pi, sqrt

class fista_solver:

    def __init__(self, fwd_op, backtracking=False):

        mv = lambda z : fwd_op.adjoint(fwd_op.matmul(z))
        self.Normal = sla.LinearOperator((fwd_op.shape[1], fwd_op.shape[1]), 
                                    matvec=mv)
        
        self.lip = 2*sla.eigs(self.Normal, k=1)
        self.fwd_op = fwd_op
        self.backtracking = backtracking

        if self.backtracking:
            raise NotImplementedError("FISTA with backtracking is not implemented yet. Please use FISTA with constant stepsize.")

    def solve(self, y_meas,
            x_init=np.array([None]),
            t_init=1.,
            n_iter=100,
            verbose=True,
            verbose_rate=10,
            stop_criterion=None,
            ):

        if self.backtracking:
            pass

        if x_init.all()==None:
            x_init = np.zeros(self.fwd_op.shapeOI[1], dtype=complex)

        y = x_init.copy()
        x = x_init.copy()

        for i in range(n_iter):

            xnext = utils.soft(y, 1./self.lip)

            tnext = 0.5 * (1 + sqrt(1+4*t**2))

            y = x + (t-1)/tnext * (xnext - x)

            t = tnext
            x = xnext

            

        
