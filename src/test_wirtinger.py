""" for reloading this file after updating, use the following:
`>>> import importlib`
`>>> import test_wirtinger`
`>>> from test_wirtinger import *`
`>>>` <do some coding here>
`>>> importlib.reload(test_wirtinger);from test_wirtinger import *`
"""

import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import rescale
from numpy import pi, cos, sin
from scipy.special import gamma as special_gamma
import os
import scipy.io

import utils

from mask import *
# from fienup import *
from wirtinger import *


class trial_forward_operator:

    def __init__(self, m,n):
        self.value = np.random.randn(m,n) + 1.j*np.random.randn(m,n)
        self.shape = (m,n)
        self.shapeOI = (m,n)


    def apply(self, z):
        A = self.value
        return A @ z


    def adj_apply(self, y):
        A = self.value
        return A.T.conj() @ y


    def spectral(self, I, z, mask=None):

        m = self.shape[0]
        print(m)

        return (1./m) * self.adj_apply( I * self.apply(z) )


# tests

def test_adj():

    N = 256
    open_rate = 0.5
    num_masks = 10

    # define mask
    mask = masks(num_masks, (N,N))
    mask.create(open_rate=open_rate, one_aperture_size=64, Type='phase_binary')

    # define forward operator
    fwd_op = forward_op(mask=mask)

    a = np.random.randn(*fwd_op.shapeOI[1]) + 1.j*np.random.randn(*fwd_op.shapeOI[1])
    b = np.random.randn(*fwd_op.shapeOI[0]) + 1.j*np.random.randn(*fwd_op.shapeOI[0])

    inprod1 = np.sum( fwd_op.apply(a) * b.conj() )
    inprod2 = np.sum( a * fwd_op.adj_apply(b).conj())

    print(np.allclose(inprod1, inprod2))
    print(inprod1, inprod2)


def artificial_data_gaussian_fwdop():

    print('artificial_data_gaussian_fwdop')
    N0 = 128
    N = 32
    SNR = 5

    image = imread(data_dir + "/SEM_fiber.png", as_gray=True)
    c = rescale(image, scale=(N0/image.shape[0], N0/image.shape[1]), mode='reflect')
    c = c - np.amin(c)
    c = np.sqrt(c)
    c = c - np.mean(c)
    # c_phase = x**2 + y**2
    # c = c * np.exp(1j*c_phase)

    x = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(c)))
    x2d = x[::N0//N, ::N0//N]
    x = x2d.flatten()

    # define random forward operator
    fwd_op = trial_forward_operator(10*N**2, N**2)

    y = fwd_op.apply(x)
    I = abs(y)**2
    I_meas = utils.add_noise(I, SNR, 'salt_pepper')

    # approximating the frobenius norm of A with its expected value:
    n = fwd_op.shape[1]
    frob = 2 * np.sqrt(np.sqrt(n**2 + n/2 + 1/8))
    lamda = np.sqrt( fwd_op.shape[1] * np.sum(I_meas) ) / frob  
    x_init, x_est, y_est = wirtinger_flow(I_meas, fwd_op, lamda, 
        z_init='spectral', 
        mu0=0.08, 
        n_iter=100, 
        adaptive_step=True, 
        tau0 = 20,
        verbose=True)

    x_init = x_init.reshape(x2d.shape)
    x_est = x_est.reshape(x2d.shape)

    print(utils.phase_dist(x_est, x2d))


def artificial_data_coded_apertures():

    print('artificial_data_coded_apertures')

    N0 = 256
    N = 256
    SNR = 10
    num_masks = 10
    open_rate = 0.5

    image = imread(data_dir + "/SEM_fiber.png", as_gray=True)
    c = rescale(image, scale=(N0/image.shape[0], N0/image.shape[1]), mode='reflect')
    c = c - np.amin(c)
    c = np.sqrt(c)
    c = c - np.mean(c)
    # c_phase = x**2 + y**2
    # c = c * np.exp(1j*c_phase)

    x = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(c)))
    x2d = x[::N0//N, ::N0//N]
    # x = x2d.flatten()
    x = x2d


    # define mask
    mask = masks(num_masks, (N,N))
    mask.create(open_rate=open_rate, one_aperture_size=64, Type='phase_binary')

    # define forward operator
    fwd_op = forward_op(mask=mask)

    y = fwd_op.apply(x)
    I = abs(y)**2
    I_meas = utils.add_noise(I, SNR, 'salt_pepper')

    # approximating the frobenius norm of A with its expected value:
    n = fwd_op.shape[1]

    frob = np.sqrt( fwd_op.shape[0]) * np.sqrt(fwd_op.shape[1] )
    # frob = np.sqrt(fwd_op.shape[0])
    lamda = np.sqrt( fwd_op.shape[1] * np.sum(I_meas) ) / frob  
    # lamda = la.norm(x2d)

    # Optimal parameters for image size 128x128:
    # mu = 1.2
    # n_iter = 200
    # adaptive_step = True
    # tau0 = 500

    x_init, x_est, y_est = wirtinger_flow(I_meas, fwd_op, lamda, 
        z_init='spectral', 
        mu0=0.001, 
        n_iter=200, 
        adaptive_step=True, 
        tau0 = 1500,
        verbose=True)


    # Following values are optimal for a ternary coded aperture.
    # x_init, x_est, y_est = wirtinger_flow(I_meas, fwd_op, lamda, 
    #   z_init='spectral', 
    #   mu0=1.2, 
    #   n_iter=200, 
    #   adaptive_step=True, 
    #   tau0 = 100,
    #   verbose=True)

    x_init = x_init.reshape(x2d.shape)
    x_est = x_est.reshape(x2d.shape)

    print(utils.phase_dist(x_est/la.norm(x_est), x2d/la.norm(x2d)))
    print(la.norm(x_init))
    print(la.norm(x2d))
    # plt.figure(1);plt.imshow(np.angle(x_init));plt.colorbar()
    plt.figure(2);plt.imshow(np.angle(y)[0,:,:]);plt.colorbar()
    plt.figure(3);plt.imshow(np.angle(y_est)[0,:,:]);plt.colorbar()
    plt.show()


def artificial_data_coded_apertures_with_real_input():

    print('artificial_data_coded_apertures_with_real_input')
    N = 256
    SNR = np.inf
    num_masks = 10
    open_rate = 0.5

    data_dir = "C:/Users/vak2/OneDrive for Business/turbid/Experiments/CSHG/500nm/190507/data/"

    masterp = scipy.io.loadmat(data_dir+"masterp.mat")['masterp_save']

    x = backpropagator(masterp.reshape((1,*masterp.shape)))  # this is the object

    mask_filenames = []
    for file in os.listdir(data_dir+"maskth/"):
        if file.endswith(".tif"):
            mask_filenames += [data_dir+"maskth/"+file]


    mask = masks(num_masks, (N,N))

    assert len(mask_filenames)==num_masks, "Number of masks does not match number of files"
    mask.get_mask(mask_filenames)
    # rescale mask between -1 and 1
    mask.value = 2 * (mask.value/255.) - 1
    mask.value = -mask.value

    # define forward operator 
    fwd_op = forward_op(mask=mask)

    y = fwd_op.apply(x)
    I = abs(y)**2
    I_meas = utils.add_noise(I, SNR, 'salt_pepper')

    # approximating the frobenius norm of A with its expected value:
    n = fwd_op.shape[1]

    frob = np.sqrt(fwd_op.shape[0]) * np.sqrt(fwd_op.shape[1] )
    lamda = np.sqrt( fwd_op.shape[1] * np.sum(I_meas) ) / frob  
    # lamda = la.norm(x2d)

    # Optimal parameters for image size 128x128:
    # mu = 1.2
    # n_iter = 200
    # adaptive_step = True
    # tau0 = 500

    x_init, x_est, y_est = wirtinger_flow(I_meas, fwd_op, lamda, 
        z_init='spectral', 
        mu0=0.001, 
        n_iter=200, 
        adaptive_step=True, 
        tau0 = 1500,
        verbose=True)


    # Following values are optimal for a ternary coded aperture.
    # x_init, x_est, y_est = wirtinger_flow(I_meas, fwd_op, lamda, 
    #   z_init='spectral', 
    #   mu0=1.2, 
    #   n_iter=200, 
    #   adaptive_step=True, 
    #   tau0 = 100,
    #   verbose=True)

    # x_init = x_init.reshape(x2d.shape)
    # x_est = x_est.reshape(x2d.shape)

    # print(utils.phase_dist(x_est/la.norm(x_est), x2d/la.norm(x2d)))
    # print(la.norm(x_init))
    # print(la.norm(x2d))
    # # plt.figure(1);plt.imshow(np.angle(x_init));plt.colorbar()
    # plt.figure(2);plt.imshow(np.angle(y)[0,:,:]);plt.colorbar()
    # plt.figure(3);plt.imshow(np.angle(y_est)[0,:,:]);plt.colorbar()
    # plt.show()


def real_data_coded_apertures(niter):
# if True:

    niter = 500

    print('real_data_coded_apertures')
    N = 256
    num_masks = 10
    mode='amplify'

    # data directory
    data_dir = "C:/Users/vak2/OneDrive for Business/turbid/Experiments/CSHG/500nm/190508/data/"

    mask_filenames = []
    for file in os.listdir(data_dir+"maskth/"):
        if file.endswith(".tif"):
            mask_filenames += [data_dir+"maskth/"+file]


    mask = masks(num_masks, (N,N))

    assert len(mask_filenames)==num_masks, "Number of masks does not match number of files"
    mask.get_mask(mask_filenames)

    # rescale mask between -1 and 1
    mask.value = 2 * (mask.value/255.) - 1
    mask.value = -mask.value

    # put mask boundaries to zero (to account for phase singularities)
    # Estimate statistics for calculating the frobenius norm from the emperical probability mass distribution on {-1,0,1} in the masks
    mask.nullify_boundaries()
    expected = 0.6101730346679688

    # define forward operator 
    fwd_op = forward_op(mask=mask, mode=mode)

    # get measured data
    I_meas = scipy.io.loadmat(data_dir+"meas.mat")['meas_save']
    I_meas = np.transpose(I_meas, (2,0,1))
    I_meas[I_meas<0.] = 0.
    I_meas = I_meas**2
    I_meas = I_meas[:,::-1,::-1]

    master = scipy.io.loadmat(data_dir+"masterp.mat")['masterp_save']
    master = master[::-1,::-1]
    master[master<0.] = 0.

    # approximating the frobenius norm of A with its expected value:
    n = fwd_op.shape[1]

    if mode=='amplify':
        frob = np.sqrt(fwd_op.shape[0]) * np.sqrt(fwd_op.shape[1] )
    elif mode=='ortho':
        frob = np.sqrt(fwd_op.shape[0])

    frob = expected * frob 
    lamda = np.sqrt( fwd_op.shape[1] * np.sum(I_meas) ) / frob

    x_init, x_est, y_est, final_loss = wirtinger_flow(I_meas, fwd_op, lamda, 
            z_init='spectral', 
            mu0=0.01, 
            n_iter=niter, # 500
            adaptive_step=True, 
            tau0 = 10000,
            tikhonov=True,
            reg=0.001,
            include_gersh=True,
            gersh_proj=master,
            verbose=True)

    return x_init, x_est, fwd_op, I_meas, final_loss

    # plt.imshow(np.abs(x_init));plt.colorbar();plt.show()

def overnight_run_with_high_hopes(niters):
    x_inits = []
    x_ests = []
    fwd_ops = []
    Is = []
    losses = []
    # niters = [10, 300, 500, 1000, 1500, 2000, 2500, 3000]

    for niter in niters:
        x_init, x_est, fwd_op, I_meas, loss = real_data_coded_apertures(niter)
        x_inits = x_inits + [x_init]
        x_ests = x_ests + [x_est]
        fwd_ops = fwd_ops + [fwd_op]
        Is = Is + [I_meas]
        losses = losses + [loss]

    return x_inits, x_ests, fwd_ops, Is, losses, niters

