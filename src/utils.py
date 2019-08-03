import numpy as np 
import scipy.linalg as la 


def add_noise(x, SNR, mode='gaussian'):
	""" Adds gaussian noise of a given SNR to a signal
	"""

	p_signal = la.norm(x)**2
	snr_inv = 10**(-0.1*SNR)

	p_noise = p_signal * snr_inv 
	sigma = np.sqrt( p_noise/np.prod(x.shape) )

	if mode=='gaussian':
		x_noisy = x + sigma * np.random.randn(*(x.shape))
	if mode=='salt_pepper':
		x_noisy = x + sigma * abs(np.random.randn(*(x.shape)))

	return x_noisy

def phase_dist(xa,xb):
	""" Should be 0 if x and y differ only by a global phase.
	"""
	x1 = xa.flatten()
	x2 = xb.flatten()

	return np.sqrt( la.norm(x1)**2 + la.norm(x2)**2 - 2*abs(np.vdot(x1,x2)) )