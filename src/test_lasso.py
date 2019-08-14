import numpy as np 
import matplotlib.pyplot as plt 
# plt.rcParams.update({'font.size': 22})
from PIL import Image
import importlib
import os
import sys

from lasso import *
from forward_model import *
from mask import *
import utils
sys.path.append("../Curvelab/fdct3d/src/")
import pyfdct3d
import pywt


def test_wavelet_sparsity():

    N = 256
    NZ = 10
    data_dir = "../data/"
    filename = "phantom2d.tif"
    alpha = 70.
    sigma = 5.
    num_masks = 10

    WAVELET = pywt.Wavelet('coif3')
    LEVEL = 2

    im = Image.open(data_dir+filename)
    xgt = np.zeros((N,N,NZ), dtype=complex)
    xgt[...,0] = np.array(im).astype(complex)[::2,::2]
    xgt = xgt / abs(xgt).max()

    for i in range(1,NZ):
        xgt[:,:,i] = utils.elastic_distortion(xgt[:,:,0], alpha, sigma)

    c,_ = pywt.coeffs_to_array(pywt.wavedecn(xgt, wavelet=WAVELET, level=LEVEL))
    c = c.flatten()
    print(c.shape)

    plt.hist(abs(c), bins=1000)
    plt.yscale('log', nonposy='clip')
    plt.show()


def test_curvelet_sparsity():

    NBSCALES = 4
    NBDSTZ_COARSE = 2
    AC = 0

    N = 256
    NZ = 10
    data_dir = "../data/"
    filename = "phantom2d.tif"
    alpha = 70.
    sigma = 5.
    num_masks = 10

    im = Image.open(data_dir+filename)
    xgt = np.zeros((N,N,NZ), dtype=complex)
    xgt[...,0] = np.array(im).astype(complex)[::2,::2]
    xgt = xgt / abs(xgt).max()

    for i in range(1,NZ):
        xgt[:,:,i] = utils.elastic_distortion(xgt[:,:,0], alpha, sigma)
        xgt[:,:,i] = xgt[:,:,i]+xgt[:,::-1,i]+xgt[::-1,:,i]

    params = pyfdct3d.get_curvelet_params(xgt.shape, NBSCALES, NBDSTZ_COARSE, AC)
    c = pyfdct3d.curvedec3(xgt, params)
    xest = pyfdct3d.curverec3(c, params)
    print(np.allclose(xest, xgt))
    print(params)

    plt.hist(abs(c), bins=1000)
    plt.yscale('log', nonposy='clip')
    plt.show()



# def test_lasso_artificial_data_real_masks():
if True:

    N = 256
    NZ = 10
    data_dir = "../data/"
    filename = "phantom2d.tif"
    alpha = 70.
    sigma = 5.
    num_masks = 10
    type_masks = "ex"
    SNR = 20

    im = Image.open(data_dir+filename)
    xgt = np.zeros((N,N,NZ), dtype=complex)
    xgt[...,0] = np.array(im).astype(complex)[::2,::2]
    xgt = xgt / abs(xgt).max()

    for i in range(1,NZ):
        xgt[:,:,i] = utils.elastic_distortion(xgt[:,:,0], alpha, sigma)
    
    # get mask from mask files
    mask_filenames = []
    mask_dir = data_dir+"tissue/maskth/"
    for file in os.listdir(mask_dir):
        if file.endswith(".tif"):
            mask_filenames += [mask_dir+file]

    mask = masks(num_masks, (N,N))

    if type_masks=="ex":
        assert len(mask_filenames)==num_masks, "Number of masks does not match number of files"
        mask.get_mask(mask_filenames)
        # rescale mask between -1 and 1
        mask.value = 2 * (mask.value/255.) - 1
        mask.value = -mask.value
    else:
        mask.create(open_rate=0.5, one_aperture_size=32, Type='phase_binary')

    fwd_op = ForwardModel(xgt.shape, mask, include_shg=True)

    # # generate artificial data
    y_true = fwd_op(xgt)
    y_meas = utils.add_noise(y_true, SNR, mode='complex')
    y_hol = fwd_op.uscope(xgt)
    y_hol_meas = utils.add_noise(y_hol, SNR, mode='complex')
    x_init = fwd_op.adjoint(y_meas)/mask.num/NZ

    solver = LassoSolver(fwd_op, use_fista=False)
    # xest,_,_ = solver.solve_fista(y_meas, n_iter=100,
    #                    sparsifying="curvelets",
    #                    print_recon=True,
    #                    ground_truth=xgt)
    xest = solver.solve_ista(y_meas,
                            x_init=x_init,
                            step=1.e-2,
                            lam=1.e-3,
                            n_iter=500,
                            step_scheduling=0.9999,
                            reg_scheduling=0.995,
                            sparsifying="curvelets",
                            print_recon=True,
                            ground_truth=xgt)

    print("Reconstruction Mean Squared Error :", utils.mse(xgt, xest))
    print("Reconstruction PSNR :", utils.psnr(xgt, xest))
    print("Reconstruction Relative Error :", utils.rel_err(xgt, xest), "%")

    xhol = holographic_recon(fwd_op.uscope, y_hol_meas)
    print("Holo. Recon. Mean Squared Error :", utils.mse(xgt, xhol))
    print("Holo. Recon. PSNR :", utils.psnr(xgt, xhol))
    print("Holo. Recon. Relative Error :", utils.rel_err(xgt, xhol), "%")

def imageplot(i):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(abs(xest[:,:,i]))
    plt.subplot(122)
    plt.imshow(abs(xgt[:,:,i]))
    plt.show()
