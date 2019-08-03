import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import eigs as sparse_eigs
import utils

from numpy import pi, cos, sin, sqrt

from mask import *


def propagator2d(x, mode='amplify'):
	""" Forward propagator based on an optics kernel. (lens fourier transform in this case)
	"""
	if mode=='amplify':
		y = np.prod(x.shape) * np.fft.ifft2(np.fft.ifftshift(x))
	elif mode=='ortho':
		y = np.fft.ifft2(np.fft.ifftshift(x), norm='ortho')

	return y


def backpropagator2d(y, mode='amplify'):
	""" Backward propagator based on an optics kernel. (lens fourier transform in this case)
	"""
	if mode=='amplify':
		x = np.fft.fft2(y)
	elif mode=='ortho':
		x = np.fft.fft2(y, norm='ortho')

	return np.fft.fftshift(x)


def propagator(x, mode='amplify'):
	""" Forward propagator based on an optics kernel. (lens fourier transform in this case)
	"""
	if mode=='amplify':
		y = np.prod(x.shape[1::]) * np.fft.ifft2(np.fft.ifftshift(x, axes=(1,2)))
	elif mode=='ortho':
		y = np.fft.ifft2(np.fft.ifftshift(x, axes=(1,2)), norm='ortho')

	return y


def backpropagator(y, mode='amplify'):
	""" Backward propagator based on an optics kernel. (lens fourier transform in this case)
	"""
	if mode=='amplify':
		x = np.fft.fft2(y)
	elif mode=='ortho':
		x = np.fft.fft2(y, norm='ortho')

	return np.fft.fftshift(x, axes=(1,2))


class forward_op:

	""" Forward operator, from just behind the coded aperture to just in front of the detector plane.

	Attributes:
	`mask` : `mask` of object of class `masks`

	Methods:
	`apply`     : Apply the forward operator to an input.
	`adj_apply` : Apply the adjoint of the foward operator to an input
	`spectral`  : Computes the action of a spectral operator (1/m) A* diag(I) A. Leading eigenvector of this gives the spectral estimate.

	"""


	def __init__(self, mask=None, mode='amplify'):
		self.mask = mask
		self.shape = (np.prod(mask.shape) * mask.num, np.prod(mask.shape))
		self.shapeOI = ( (mask.num, *mask.shape), mask.shape )
		self.mode = mode


	def apply(self, z, mask=None):

		""" Apply the forward operator to an input.
		Inputs :
		`z`    : Input vector
		`mask` : Coded aperture mask. If `None` is specified, value stored in `self.mask` will be used.

		Outputs :
		`y`    : Electric field at the detector plane.
		"""

		if mask==None:
			mask = self.mask

		zm = mask.apply(z)
		y = propagator(zm, mode=self.mode)

		return y


	def adj_apply(self, y, mask=None):

		""" Apply the adjoint of forward operator to an input.
		Inputs :
		`y`    : Input vector
		`mask` : Coded aperture mask. If `None` is specified, value stored in `self.mask` will be used.

		Outputs :
		`z`    : computed action of adjoint
		"""
		num_masks = self.mask.num

		if mask==None:
			mask = self.mask

		# xm = np.prod(y.shape[1::]) * backpropagator(y)
		xm = backpropagator(y, mode=self.mode)
		x = mask.apply(xm)

		return np.sum(x, axis=0)


def spectral(fwd_op, I, z, mask=None):

	""" Computes the action of a spectral operator (1/m) A* diag(I) A. Leading eigenvector of this gives the spectral estimate.

	Inputs : 
	`I`    : Intensity measurements
	`mask` : `mask` object

	Output : 
	``action of the spectral operator
	"""
	m = fwd_op.shape[0]

	return (1./m) * fwd_op.adj_apply( I * fwd_op.apply(z) )
	# return fwd_op.adj_apply( I * fwd_op.apply(z) )


def wirtinger_grad(z, I, fwd_op):

	""" Wirtinger gradient for wirtinger flow gradient descent.

	Inputs : 
	`z`      : Point at which to evaluate the wirtinger gradient
	`I` 	 : Measured intensity
	`fwd_op` : Forward operator with pre-loaded mask.

	Outputs : 
	`w` 	 : Wirtinger flow derivative.
	"""

	m = fwd_op.shape[0]

	t1 = fwd_op.apply(z)
	t2 = np.abs(t1)**2 - I

	t3 = fwd_op.adj_apply( t2 * t1 )

	return 1/m * t3


def spectral_est(fwd_op, I):

	""" Computes the spectral estimate for initializing the wirtinger gradient descent.

	Inputs :
	`fwd_op` : Forward operator with pre-loaded mask.
	`I`		 : Measured instensity

	Outputs : 
	`z0`     : Spectral estimate : The leading eigenvector of (1/m) sum_{r=1}^m a_r a_r* y_r. a_r are the columns of A.H. 

	"""

	matrix_shape = fwd_op.shape
	
	mv = lambda z: spectral(fwd_op, I, z.reshape(fwd_op.shapeOI[1]), mask=None).reshape((fwd_op.shape[1],))
	print(matrix_shape)

	Y = sla.LinearOperator((matrix_shape[1],matrix_shape[1]), matvec=mv)
	# Y = sla.LinearOperator(matrix_shape)

	eigval, eigvec = sparse_eigs(Y, k=1)

	eigvec = eigvec.reshape(fwd_op.shapeOI[1])

	m = fwd_op.shape[0]

	z0 = eigvec / m

	return z0


def wirtinger_flow(I, fwd_op, lamda, z_init='spectral', mu0=0.4, n_iter=100, 
	adaptive_step=True, 
	tau0 = None,
	tikhonov=False,
	reg=None,
	include_gersh=False,
	gersh_proj=None,
	verbose=True):

	""" Phase retrieval using wirtinger flow using a spectral initialization.

	Inputs : 
	`I`      		: Measured intensity. (Vector formed out of |<a_i, z>|^2 measurements)
	`fwd_op` 		: Forward opeartor. This is an object of class `forward_op`. Attributes contain `mask`, methods contain action of the forward operator, action of the adjoint and action of the spectral operator.
	`x_init` 		: Initialization. default is the `spectral` for the spectral estimate.
	`mu`     		: Learning rate
	`n_iter`        : Number of iterations
	`adaptive_step` : If `True`, an adaptive step size is used, with an asymptotically increasing step size [1].
	`tau0`   		: Parameter controlling the adaptive learning rate. If `None`, sets a default value equal to
	`verbose`		: Boolian. If `True`, prints the loss function at every 10th iteration (can change this)

	Outputs :
	Final estimate at the input of `fwd_op` as well as the detector plane.

	Heuristics for hyperparameter tuning :
	As stated in 
	"""

	if adaptive_step==True and tau0==None:
		tau0 = 330

	if tikhonov==True and reg==None:
		reg = 0.001


	if z_init=='spectral':
		z_init = spectral_est(fwd_op, I)

		# normalization constant for z_init:
		z_init = (z_init / la.norm(z_init)) * lamda 	

	z = z_init.copy()

	normz02 = la.norm(z_init)**2
	m = fwd_op.shape[0]

	try:
		for i in range(n_iter):

			if adaptive_step==True:
				mu = min( 1-np.exp(-(i+1)/tau0), mu0)

			z = z - (mu/normz02) * wirtinger_grad(z, I, fwd_op)

			if tikhonov==True:
				z = z - reg * (mu/normz02) * z

			if verbose==True and n_iter%10==0:
				loss = la.norm(I - abs(fwd_op.apply(z))**2)**2
				print('Loss :', loss)

			if include_gersh==True and i>0.8*n_iter and i%5==0:
				zr = propagator2d(z)
				zr = la.norm(zr)/la.norm(gersh_proj) * gersh_proj / abs(zr) * zr
				z = 1./np.prod(zr.shape) * backpropagator2d(zr)


		z_est = z
		y_est = fwd_op.apply(z_est)

		if verbose==True:
			final_loss = loss
		else:
			final_loss = None

		return z_init, z_est, y_est, final_loss


	except (OverflowError, ValueError):
		print('Just returning the Spectral Estimate')
		return z_init, None, None, None




