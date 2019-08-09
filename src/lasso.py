import numpy as np 
import scipy.linalg as la 
import scipy.sparse.linalg as sla

import utils
from forward_model import *
from physical_params import *
from mask import *

from numpy import cos, sin, pi, sqrt

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

    def solve_fista(self, z_meas,
            x_init=np.array([None]),
            t_init=1.,
            n_iter=100,
            verbose=True,
            verbose_rate=10,
            stop_criterion=None,
            sparsifying=None,
            ):
        """ Solves lasso using fista 
        """

        if self.backtracking:
            pass

        if x_init.all()==None:
            x_init = np.zeros(self.fwd_op.shapeOI[1], dtype=complex)

        y = x_init.copy()
        x = x_init.copy()
        step = 1./self.lip
        fwd_op = self.fwd_op

        for i in range(n_iter):

            y = y -  step * fwd_op.adjoint(fwd_op(y) - z_meas)

            if sparsifying==None:
                xnext = utils.soft(y, 1./self.lip)
            elif sparsifying=="wavelets":
                xnext = utils.soft_wavelets(y, 1./self.lip)
            elif sparsifying=="curvelets":
                xnext = utils.soft_curvelets(y, 1./self.lip)

            tnext = 0.5 * (1 + sqrt(1+4*t**2))

            y = x + (t-1)/tnext * (xnext - x)

            t = tnext
            x = xnext

            if verbose and i%verbose_rate==0:
                loss = la.norm(fwd_op(x)-z_meas)**2 
                print("Iter :", i, ", Loss :", loss)

            if stop_criterion!=None:

                err_prev = err
                err = la.norm(fwd_op(x)-z_meas)
                
                if abs(err-err_prev)/err < stop_criterion:
                    return x, y, t

        return x, y, t

                

        
