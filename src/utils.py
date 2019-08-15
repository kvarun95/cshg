import sys 
import numpy as np 
import scipy.linalg as la 
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# for wavelet transforms
import pywt 
WAVELET = pywt.Wavelet('coif3')
LEVEL = 2

# for curvelet transforms
sys.path.append("../Curvelab/fdct3d/src/")
try:
	import pyfdct3d
except ImportError:
	pass
NBSCALES = 4 # number of scales of the curvelet transform
NBDSTZ_COARSE = 3
AC = 0 # wavelet or curvelet initialization

from numpy import pi, cos, sin, exp


def add_noise(x, SNR, mode='gaussian'):
	""" Adds gaussian noise of a given SNR to a signal
	"""

	p_signal = la.norm(x)**2
	snr_inv = 10**(-0.1*SNR)

	p_noise = p_signal * snr_inv 
	sigma = np.sqrt( p_noise/np.prod(x.shape) )

	if mode=='gaussian':
		x_noisy = x + sigma * np.random.randn(*(x.shape))
	elif mode=='salt_pepper':
		x_noisy = x + sigma * abs(np.random.randn(*(x.shape)))
	elif mode=='complex':
		x_noisy = x + sigma/np.sqrt(2) * (np.random.randn(*(x.shape)) + 1.j*np.random.randn(*(x.shape)))
	else:
		raise ValueError("Enter a suitable mode")

	return x_noisy


def phase_dist(xa,xb):
	""" Should be 0 if x and y differ only by a global phase.
	"""
	x1 = xa.flatten()
	x2 = xb.flatten()

	return np.sqrt( la.norm(x1)**2 + la.norm(x2)**2 - 2*abs(np.vdot(x1,x2)) )


def cis(theta):

	return cos(theta) + 1.j*sin(theta)


def elastic_distortion(image, alpha, sigma):
	
	shape = image.shape
	dx = alpha * gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)
	dy = alpha * gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)

	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
	indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

	distorted_image_real = map_coordinates(image.real, indices, order=1, mode='reflect')
	distorted_image_imag = map_coordinates(image.imag, indices, order=1, mode='reflect')
	return distorted_image_real.reshape(image.shape) + 1.j*distorted_image_imag.reshape(image.shape)


def soft(x, lamda):
	""" `x` : Complex numpy ndarray
	    `lamda` : Real Scalar
	"""
	are = abs(x.real)/abs(x)
	aim = abs(x.imag)/abs(x)

	zreal = (x.real - lamda*are)*(x.real > lamda*are) + (x.real + lamda*are)*(x.real < -lamda*are) 
	zimag = (x.imag - lamda*aim)*(x.imag > lamda*aim) + (x.imag + lamda*aim)*(x.imag < -lamda*aim)	

	return zreal + 1.j*zimag


def soft_wavelets(x, lamda):

	# coeffs = pywt.wavedecn(x, wavelet=WAVELET, level=LEVEL, axes=(0,1) )
	coeffs = pywt.wavedecn(x, wavelet=WAVELET, level=LEVEL)
	# c, slices = pywt.coeffs_to_array(coeffs, axes=(0,1))
	c, slices = pywt.coeffs_to_array(coeffs)
	c = soft(c, lamda)
	coeffs = pywt.array_to_coeffs(c, slices, output_format='wavedecn')

	# return pywt.waverecn(coeffs, WAVELET, axes=(0,1))
	return pywt.waverecn(coeffs, WAVELET)


def curvelet_params(input_shape):

	return pyfdct3d.get_curvelet_params(input_shape, NBSCALES, NBDSTZ_COARSE, AC)


def soft_curvelets(x, lamda, params):

	return pyfdct3d.curvesoft3(x, lamda, params)


def soft_curvelets2(x, lamda, params):

	xlist = [pyfdct3d.curvesoft3(x[:,:,i:i+1], lamda, params) for i in range(x.shape[2])]
	return np.concatenate(xlist, axis=2)


def mse(x1, x2):
	return la.norm(x1-x2)**2/np.prod(x1.shape)


def psnr(x1, x2):
	recon_mse = mse(x1,x2)
	return 10*np.log10(abs(x1).max()**2/recon_mse)


def rel_err(x1, x2):
	return la.norm(x1-x2)/la.norm(x1)