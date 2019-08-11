import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 22})
from PIL import Image
import importlib

from forward_model import *

def test_fresnelprop():
# if True:

    N = 256
    NZ = 10
    data_dir = "../data/"
    filename = "phantom2d.tif"

    im = Image.open(data_dir+filename)
    xgt = np.zeros((N,N,NZ), dtype=complex)
    xgt[...,0] = np.array(im).astype(complex)[::2,::2]
    xgt = xgt / abs(xgt).max()
    # plt.imshow(xgt.real);plt.colorbar()
    # plt.show()

    fwd_op = FresnelProp(input_shape=(N,N,NZ), output_shape=(N,N), include_shg=True)
    y1 = fwd_op(xgt)

    fwd_op = FresnelProp(input_shape=(N,N,NZ), output_shape=(N,N), include_shg=False)
    y2 = fwd_op(xgt)


# test the adjoint
def test_fresnelprop_adj():
# if True:

    N = 256
    NZ = 10

    fwd_op = FresnelProp(input_shape=(N,N,NZ), output_shape=(N,N), include_shg=True)

    x1 = np.random.rand(N,N,NZ) + 1.j*np.random.rand(N,N,NZ)
    y1 = fwd_op(x1)

    y2 = np.random.rand(N,N) + 1.j*np.random.rand(N,N)
    x2 = fwd_op.adjoint(y2)

    ip1 = np.vdot(y1, y2)
    ip2 = np.vdot(x1, x2)
    print(np.allclose(ip1, ip2))    


def test_fourierprop_adj():
# if True:

    N = 256
    open_rate = 0.5
    num_masks = 10

    mask = masks(num_masks, (N,N))
    mask.create(open_rate=open_rate, one_aperture_size=64, Type='phase_binary')
    fwd_op = FourierProp(mask)

    x1 = np.random.rand(N,N) + 1.j*np.random.rand(N,N)
    y1 = fwd_op(x1)

    y2 = np.random.rand(num_masks, N,N) + 1.j*np.random.rand(num_masks, N,N)
    x2 = fwd_op._adjoint(y2)

    ip1 = np.vdot(y1, y2)
    ip2 = np.vdot(x1, x2)
    print(np.allclose(ip1, ip2))    