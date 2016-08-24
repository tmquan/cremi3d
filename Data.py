import h5py
import numpy as np
import skimage.io
import scipy

from scipy import ndimage
from cremi import Annotations, Volume
from cremi.io import CremiFile
from Config import *
###################################################################################
def Reading(filename, isTest=False):
	# # Read the data into dataset
	# print "Filename: ", filename
	# # with h5py.File('sample_A_20160501.hdf', 'r') as f:
	# with h5py.File(filename, 'r') as f:
		# print f["volumes"]
		# imageDataSet = f["volumes/raw"][:]
		# labelDataSet = f["volumes/labels/neuron_ids"][:]
	
	# imageDataSet = imageDataSet.astype(np.float32)
	# labelDataSet = labelDataSet.astype(np.float32)
	# return imageDataSet, labelDataSet
	file = CremiFile(filename, "r")
	print filename
	# Check the content of the datafile
	print "Has raw			: " + str(file.has_raw())
	print "Has neuron ids	: " + str(file.has_neuron_ids())
	print "Has clefts		: " + str(file.has_clefts())
	print "Has annotations	: " + str(file.has_annotations())

	
	# Read everything there is.
	#
	# If you are using the padded versions of the datasets (where raw is larger to 
	# provide more context), the offsets of neuron_ids, clefts, and annotations tell 
	# you where they are placed in nm relative to (0,0,0) of the raw volume.
	#
	# In other words, neuron_ids, clefts, and annotations are exactly the same 
	# between the padded and unpadded versions, except for the offset attribute.
	raw = file.read_raw()
	if not isTest:
		neuron_ids = file.read_neuron_ids()
		clefts = file.read_clefts()
		annotations = file.read_annotations()
	
	
	print "Read raw: 	" 	+ str(raw) + \
		  ", resolution " 	+ str(raw.resolution) + \
		  ", offset 	" 	+ str(raw.offset) + \
		 ("" if raw.comment == None else ", comment \"" + raw.comment + "\"")
	if not isTest:
		print "Read neuron_ids: " 	+ str(neuron_ids) + \
			  ", resolution " 	  	+ str(neuron_ids.resolution) + \
			  ", offset " + str(neuron_ids.offset) + \
			 ("" if neuron_ids.comment == None else ", comment \"" + neuron_ids.comment + "\"")
		# neuron_ids.offset will contain the starting point of neuron_ids inside the raw volume. 
		# Note that these numbers are given in nm.
		
		
		# print "Read clefts: " + str(clefts) + \
		# ", resolution " + str(clefts.resolution) + \
		# ", offset " + str(clefts.offset) + \
		# ("" if clefts.comment == None else ", comment \"" + clefts.comment + "\"")
		
		# print "Read annotations:"
		# for (id, type, location) in zip(annotations.ids(), annotations.types(), annotations.locations()):
			# print str(id) + " of type " + type + " at " + str(np.array(location)+np.array(annotations.offset))
			# print "Pre- and post-synaptic partners:"
		# for (pre, post) in annotations.pre_post_partners:
			# print str(pre) + " -> " + str(post)
	with h5py.File(filename, 'r') as f:
		print f["volumes"]
		imageDataSet = f["volumes/raw"][:]
		if not isTest:
			labelDataSet = f["volumes/labels/neuron_ids"][:]
	imageDataSet = imageDataSet.astype(np.float32)
	if not isTest:
		labelDataSet = labelDataSet.astype(np.float32)	
	if not isTest:
		return imageDataSet, labelDataSet
	return imageDataSet
	
###################################################################################	
def membrane(array):
	shape  = array.shape
	print shape
	result = np.zeros(shape, dtype=array.dtype)
	array  = np.squeeze(array)
	result = ndimage.laplace(array)
	result[result !=  0] = +1.0
	result[result ==  0] = 255.0
	result[result == +1] = 0.0
	return result
###################################################################################
# def shrink_border(labels):
# def membrane(array):
	# shape  = array.shape
	# print shape
	# result = np.zeros(shape, dtype=array.dtype)
	# array  = np.squeeze(array)
	# # dx    = ndimage.sobel(image, 0)  # horizontal derivative
	# # dy    = ndimage.sobel(image, 1)  # vertical derivative
	# # dz    = ndimage.sobel(image, 2)  # vertical derivative
	# # mag   = np.hypot(dx, dy, dz)  # magnitude
	# result = scipy.ndimage.filters.generic_gradient_magnitude(array,	\
														   # scipy.ndimage.filters.sobel)
	
	# result[result !=  0] = -1.0
	# result[result ==  0] = 255.0
	# result[result == -1] = 0.0
	# return result
###################################################################################	
def WritingImTif():
	# TrainDir
	list = ['sample_A_20160501.hdf',
	        'sample_B_20160501.hdf',
	        'sample_C_20160501.hdf',
			];
	# raw, neuron_ids = Reading(trainDir + filename) # trainDir is from Config.py
	for filename in list:
		print '-'*50
		print filename
		images, labels = Reading(dataDir+filename)
				
		print images.shape
		print labels.shape
		
		# print "Saving image..."
		# skimage.io.imsave(filename + '_image' +  '.tif', images)
		# skimage.io.imsave(filename + '_label' +  '.tif', labels)
		
		# np.save(filename + '_image' +  '.npy', images)
		# np.save(filename + '_label' +  '.npy', labels)
		
		# Pad radius = (1280-1250)/2 = 15 nearest image size, pad only x and y
		# radius = 15 # See Config.py
		print "Padding image..."
		images = np.pad(images, ((radius, radius), (radius, radius), (radius, radius)), 'reflect');
		labels = np.pad(labels, ((radius, radius), (radius, radius), (radius, radius)), 'reflect');
		
						
		print images.shape
		print labels.shape
		
		## Get the membrain
		membrs = membrane(labels)
		
		print 'Membrane shape'
		print membrs.shape
		
		
		
		
		skimage.io.imsave(filename + '_image_' +  'radius_' + str(radius) + '.tif', images)
		skimage.io.imsave(filename + '_label_' +  'radius_' + str(radius) + '.tif', labels)
		skimage.io.imsave(filename + '_membr_' +  'radius_' + str(radius) + '.tif', membrs)
		
		###################################################################################
		## Process the 3D images here
		images_0 = images[radius-1:-radius-1,:,:]
		images_1 = images[radius+0:-radius+0,:,:]
		images_2 = images[radius+1:-radius+1,:,:]
		
		images_0 = np.expand_dims(images_0, axis = 1)
		images_1 = np.expand_dims(images_1, axis = 1)
		images_2 = np.expand_dims(images_2, axis = 1)
		
		images   = np.concatenate((images_0, images_1, images_2), axis=1);	
		
		## Process the 3D membrs here
		# Crop the z slice
		membrs = membrs[radius:-radius,...]
		# Expand dimension
		membrs = np.expand_dims(membrs, axis = 1)

		skimage.io.imsave(filename + '_image3d_' +  'radius_' + str(radius) + '.tif', images)
		skimage.io.imsave(filename + '_membr3d_' +  'radius_' + str(radius) + '.tif', membrs)
		
		
		np.save(filename + '_image3d_' +  'radius_' + str(radius) + '.npy', images)
		np.save(filename + '_membr3d_' +  'radius_' + str(radius) + '.npy', membrs)
		
	###############################################################################
	# # TestDir
	# # TrainDir
	# list = ['sample_AA_20160601.hdf',
	        # 'sample_BB_20160501.hdf',
	        # 'sample_CC_20160501.hdf',
			# ];
	# # raw, neuron_ids = Reading(trainDir + filename) # trainDir is from Config.py
	# for filename in list:
		# print '-'*50
		# print filename
		# images = Reading(dataDir+filename, isTest=True)
		# print images.shape
		
		# print "Saving image..."
		# skimage.io.imsave(filename + '_image' +  '.tif', images)
		# np.save(filename + '_image' +  '.npy', images)
		
		
		# # Pad radius = (1280-1250)/2 = 15 nearest image size, pad only x and y
		# # radius = 15 # See Config.py
		# print "Padding image..."
		# images = np.pad(images, ((radius, radius), (radius, radius), (radius, radius)), 'reflect');
		# print images.shape
		
		# skimage.io.imsave(filename + '_image_' +  'radius_' + str(radius) + '.tif', images)
		# np.save(filename + '_image_' +  'radius_' + str(radius) + '.npy', images)
###################################################################################	
if __name__ == '__main__':
	WritingImTif()
	
###################################################################################