import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from scipy import ndimage

class masks:

	""" defines the masks.

	Attributes:
	`num` : number of patterns
	`shape` : Shape of each 2D mask
	`value` : numpy ndarry containing the masks

	Methods:
	`create()`
	`apply()`

	"""

	def __init__(self, M, mask_shape):
		self.num = M
		self.shape = mask_shape
		self.value = np.zeros((M, *mask_shape), dtype='complex')


	def create(self, aper_size=None, open_rate=0.25, one_aperture_size=16, Type='random'):

		""" creates the mask patterns based on the keyword `type`
		Inputs:
		`aper_size` : Size of each aperture in the pattern
		`open_rate` : Ratio of open apertures to total apertures in teh mask
		`one_aperture_size` : dimensions of one aperture
		`type` : Type of coded aperture patterns to create. 
				`'random'` refers to random patterns binary amplitude patterns 
				`'nra'` refers to non-redundant array patterns
				`'nrp'` refers to non-repeating array patterns
				`'darkfield'` refers to setting the DC to zero always
				`'phase_ternary'` refers to (-1,0,1) coding with probability (0.25,0.5,0.25)
				`'phase_binary'` refers to (-1,1) coding with probability (0.5,0.5)

		"""

		M = self.num
		mask_shape = self.shape

		assert min(mask_shape)>=one_aperture_size , 'too small mask shape'

		if aper_size==None:
			aper_size = (mask_shape[0]//one_aperture_size, mask_shape[1]//one_aperture_size)

		num_apertures = (mask_shape[0]//aper_size[0], mask_shape[1]//aper_size[1])

		one_aperture = np.ones(aper_size)

		if Type=='random':
			for i in range(M):
				basic_pattern = np.random.binomial(1, open_rate, num_apertures)
				self.value[i,:,:] = np.kron(basic_pattern, one_aperture)

		elif Type=='darkfield':
			for i in range(M):
				basic_pattern = np.random.binomial(1, open_rate, num_apertures)
				basic_pattern[num_apertures[0]//2, num_apertures[1]//2] = 0
				self.value[i,:,:] = np.kron(basic_pattern, one_aperture)

		elif Type=='phase_ternary':
			for i in range(M):
				basic_pattern = np.random.choice(3, size=num_apertures, 
					p=[open_rate/2, 1-open_rate, open_rate/2] ) - 1.
				self.value[i,:,:] = np.kron(basic_pattern, one_aperture)

		elif Type=='phase_binary':
			for i in range(M):
				basic_pattern = 2*np.random.choice(2, size=num_apertures, p=[open_rate, 1-open_rate]) - 1.
				self.value[i,:,:] = np.kron(basic_pattern, one_aperture)

		elif Type=='nrp':
			pass

		else:
			raise ValueError("Undefined mask type.")

		return self.value

	def get_mask(self, filenames):
		""" Load a mask from files instead of generating it. All images should have the same shape
		"""
		assert len(filenames)<=self.num , 'too many files for upload'

		img = imread(filenames[0])#, as_gray=True)
		self.shape = img.shape

		for i in range(len(filenames)):
			img = imread(filenames[i])#, as_gray=True)
			assert img.shape==self.shape , 'Inconsistent shapes of images to be loaded'

			self.value[i,:,:] = img


	def apply(self, x):
		""" applies mask to input x
		"""

		M = self.num
		mask_shape = self.shape

		# assert x.shape==mask_shape or x.shape==(self.value).shape , 'The input does not have correct shape'

		return self.value * x


	def save(self, directory, slm_lookup=[200,238], Type='random'):
		""" saves the mask pattern as uint8 bmp file for use on slm.
		Inputs:
		`directory` : the destination folder for the patterns
		`slm_lookup`: LOW and HIGH values for SLM amplitude modulation
		"""

		M = self.num
		slm_pattern = np.array(self.shape, dtype=float)

		for i in range(M):
			if Type=='random':
				slm_pattern = self.value[i,:,:] * (slm_lookup[1]-slm_lookup[0]) + slm_lookup[0]

			if Type=='phase_ternary':
				slm_pattern = (self.value[i,:,:]==0)*slm_lookup[0] + (self.value[i,:,:]==1)*slm_lookup[1] + (self.value[i,:,:]==-1)*slm_lookup[2]

			if Type=='phase_binary':
				slm_pattern = (self.value[i,:,:]==-1)*slm_lookup[0] + (self.value[i,:,:]==1)*slm_lookup[1]

			bmpimage = Image.fromarray(slm_pattern.astype(np.uint8))
			bmpimage.save(directory+str(i)+'.bmp')

		return slm_pattern


	def nullify_boundaries(self):
		""" Calculates the boundary pixels of the mask based on the gradient value and sets them to zero.
		"""

		mask_grad = np.zeros(self.value.shape)

		for i in range(self.num):
	
			mask_grad[i,:,:] = np.sqrt( ndimage.sobel(self.value[i,:,:].real, axis=0)**2 + ndimage.sobel(self.value[i,:,:].real, axis=1)**2 )

		self.value = self.value * (mask_grad==0)

	# def estimate_stats(self):
	# 	""" Estimates the statistics and calculates the expected value of the mask multiplication factor in order to calculate the frobenius norm of the forward operator (later)
	# 	"""

	# 	h = plt.hist(self.value.real.flatten())
	# 	x = h[1][h[0]!=0]


























