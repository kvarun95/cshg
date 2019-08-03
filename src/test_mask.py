from mask import *
from skimage.io import imread
from skimage import data_dir
from skimage.transform import rescale
import importlib

def test_mask_create():

	mask = masks(3, (128,128))
	mask.create()

	plt.figure(1);plt.imshow(mask.value[0,:,:])
	plt.figure(2);plt.imshow(mask.value[1,:,:])
	plt.figure(3);plt.imshow(mask.value[2,:,:])
	plt.show()

	return mask


def test_mask_apply():

	mask = masks(3, (128,128))
	mask.create()

	image = imread(data_dir + "/phantom.png", as_gray=True)[:,:,0]
	x1 = rescale(image, scale=0.32, mode='reflect')
	x2 = np.random.rand(3, *(mask.shape))
	x2[1,:,:] = x1
	x3 = np.random.rand(5,5)

	x1m = mask.apply(x1)
	plt.figure(1);plt.imshow(x1)
	plt.figure(2);plt.imshow(x1m[1,:,:])

	x2m = mask.apply(x2)
	plt.figure(3);plt.imshow(x2[1,:,:])
	plt.figure(3);plt.imshow(x2m[1,:,:])
	plt.show()

	x3m = mask.apply(x3)


def test_mask_save(slm_lookup=[36,238]):

	mask = masks(10, (512,512))
	mask.create(open_rate=0.5, one_aperture_size=64, Type='phase_binary')
	mask.save('patterns6/', slm_lookup, Type='phase_binary')


def stripes(n):
	""" n: Number of patterns desired. Will be equally spaced between 0 to 255. 
	n is a divisor of 256.
	"""

	k = int(256/n)
	for i in range(n):

		a = np.ones((512,512))

		for j in range(10):
			a[j::20] = 238

		for j in range(10,20):
			a[j::20] = k*i

		bmpimage = Image.fromarray(a.astype(np.uint8))
		bmpimage.save('interf/interf'+str(k*i)+'.bmp')
