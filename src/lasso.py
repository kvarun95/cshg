import numpy as np 
import scipy.linalg as la 
import scipy.sparse.linalg as sla
try:
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())
except ModuleNotFoundError:
    pass

import utils
from forward_model import *
from physical_params import *
from mask import *

from numpy import cos, sin, pi, sqrt

class LassoSolver:

    def __init__(self, fwd_op, backtracking=False, use_fista=False):

        self.use_fista = use_fista

        if use_fista:
            mv = lambda z : fwd_op.adjoint(fwd_op(z.reshape(fwd_op.shapeOI[1]))).reshape(fwd_op.shape[1])
            self.Normal = sla.LinearOperator((fwd_op.shape[1], fwd_op.shape[1]), 
                                        matvec=mv) # v time consuming step
            
            self.lip = 2*sla.eigs(self.Normal, k=1)[0].real
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
            print_recon=False,
            ground_truth=None,
            ):
        """ Solves lasso using fista 
        """

        if self.backtracking:
            pass

        if x_init.all()==None:
            x_init = np.zeros(self.fwd_op.shapeOI[1], dtype=complex)

        if sparsifying=="curvelets":
            self.curve_params = utils.curvelet_params(x_init.shape)

        if print_recon:
            xgt = ground_truth

        y = x_init.copy()
        x = x_init.copy()
        step = 1./self.lip
        fwd_op = self.fwd_op
        t = t_init

        for i in range(n_iter):

            y = y -  step * fwd_op.adjoint(fwd_op(y) - z_meas)

            if sparsifying==None:
                xnext = utils.soft(y, 1./self.lip)
            elif sparsifying=="wavelets":
                xnext = utils.soft_wavelets(y, 1./self.lip)
            elif sparsifying=="curvelets":
                xnext = utils.soft_curvelets(y, 1./self.lip, self.curve_params)

            tnext = 0.5 * (1 + sqrt(1+4*t**2))

            y = x + (t-1)/tnext * (xnext - x)

            t = tnext
            x = xnext

            if verbose and i%verbose_rate==0:
                loss = la.norm(fwd_op(x)-z_meas)**2 /np.prod(z_meas.shape)
                print("Iter :", i, ", MSE :", loss)
                if print_recon:
                    recon_error = la.norm(xgt-x)**2/np.prod(x.shape)
                    print("Recon. MSE :", recon_error)

            if stop_criterion!=None:

                err_prev = err
                err = la.norm(fwd_op(x)-z_meas)
                
                if abs(err-err_prev)/err < stop_criterion:
                    return x, y, t

        return x, y, t


    def solve_ista(self, z_meas,
            x_init=np.array([None]),
            n_iter=100,
            step=1.e-3,
            lam=5.e-4,
            step_scheduling=1.,
            reg_scheduling=1.,
            verbose=True,
            verbose_rate=10,
            stop_criterion=None,
            sparsifying=None,
            print_recon=False,
            ground_truth=None):

        """ Solves lasso using Iterative Shrinkage and Thresholding.
        `x_init`         : Initial value/estimate. Default is all zeros.
        `n_iter`         : Maximum number of iterations. `int`.
        `step`           : Step size for gradient step (`float`/ `double` scalar)
        `lam`            : Regularization parameter
        `reg_scheduling` : Regularization scheduling. Multiply `lam` at each iteration by this quantity.
        `verbose`        :
        `verbose_rate`   : Print after each `verbose_rate` iterations
        `stop_criterion` :
        `sparsifying`    : Default `None`. No special sparsifying transform. 
                           If `"wavelets"`, use  wavelet transform. `pywt` package needs to be installed.
                           If `"curvelets"`, use curvelet transform. `pycfdct3d` package in `../Curvelab/fdct3d/src/` needs to be installed.
        """

        if x_init.all()==None:
            x_init = np.zeros(self.fwd_op.shapeOI[1], dtype=complex)

        if sparsifying=="curvelets":
            self.curve_params = utils.curvelet_params(x_init.shape)

        if print_recon:
            xgt = ground_truth

        x = x_init.copy()
        fwd_op = self.fwd_op

        for i in range(n_iter):

            # gradient step 
            x = x - step * fwd_op.adjoint(fwd_op(x) - z_meas)

            # proximal step
            if sparsifying==None:
                xnext = utils.soft(x, lam)
            elif sparsifying=="wavelets":
                xnext = utils.soft_wavelets(x, lam)
            elif sparsifying=="curvelets":
                xnext = utils.soft_curvelets(x, lam, self.curve_params)
            elif i==0:
                print("No sparsification used")

            # real projection
            x.imag = 0.
            # x.real[x.real<0.] = 0.

            # Regularization scheduling
            lam = reg_scheduling * lam

            # Step scheduling
            step = step_scheduling * step

            if verbose and (i%verbose_rate==0 or i==n_iter):
                loss = la.norm(fwd_op(x)-z_meas)**2 /np.prod(z_meas.shape)
                print("Iter :", i, ", MSE :", loss)
                if print_recon:
                    recon_error = la.norm(xgt-x)**2/np.prod(x.shape)
                    print("Recon. MSE :", recon_error)

            if stop_criterion!=None:

                err_prev = err
                err = la.norm(fwd_op(x)-z_meas)
                
                if abs(err-err_prev)/err < stop_criterion:
                    return x

        return x
        

def holographic_recon(fwd_op, y_meas):

    return fwd_op.adjoint(y_meas)/fwd_op.shapeOI[1][2]
        
