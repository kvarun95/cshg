import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 22})
from PIL import Image
import importlib
import os

from lasso import *
from forward_model import *
from mask import *
import utils

# def test_lasso_artificial_data_real_masks():
if True:

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
    
    # get mask from mask files
    mask_filenames = []
    mask_dir = data_dir+"tissue/maskth/"
    for file in os.listdir(mask_dir):
        if file.endswith(".tif"):
            mask_filenames += [mask_dir+file]

    mask = masks(num_masks, (N,N))

    assert len(mask_filenames)==num_masks, "Number of masks does not match number of files"
    mask.get_mask(mask_filenames)
    # rescale mask between -1 and 1
    mask.value = 2 * (mask.value/255.) - 1
    mask.value = -mask.value

    fwd_op = ForwardModel(xgt.shape, mask, include_shg=True)

    # # generate artificial data
    y_true = fwd_op(xgt)
    y_meas = utils.add_noise(y_true, 20)

    solver = LassoSolver(fwd_op, use_fista=False)
    # solver.solve_fista(y_meas, n_iter=100, sparsifying="wavelets")
    xest = solver.solve_ista(y_meas,
                            step=1.e-2,
                            lam=1.e-2,
                            n_iter=500,
                            step_scheduling=1.,
                            reg_scheduling=0.999,
                            sparsifying="wavelets")


def imageplot(i):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(abs(xest[:,:,i]))
    plt.subplot(122)
    plt.imshow(abs(xgt[:,:,i]))
    plt.show()
