from Utility import *
from Data    import *
import numpy as np
import cv2
######################################################################################
def doFlip(
	image, 
	label, 
	random_flip = np.random.choice([1, 2, 3, 4]), 
	Verbose=False
	):
	""" 
	Perform random flip
	"""
	# image = np.squeeze(image)
	# label = np.squeeze(label)

	assert((image.shape[2] == label.shape[2]) and (image.shape[1] == label.shape[1]))
	shape = image.shape
	dimk, dimy, dimx = shape

	if Verbose:
		image0 = image
		label0 = label

	flipped1 = np.zeros(image.shape)
	flipped2 = np.zeros(label.shape)
	
	# print "Label shape", label.shape
	if random_flip==1:
		for k in range(dimk):
			flipped1[k,:,:] = cv2.flip(image[k,:,:], -1)
		flipped2[0,:,:] = cv2.flip(label[0,:,:], -1)
	elif random_flip==2:
		for k in range(dimk):
			flipped1[k,:,:] = cv2.flip(image[k,:,:], 0)
		flipped2[0,:,:] = cv2.flip(label[0,:,:], 0)
	elif random_flip==3:
		for k in range(dimk):
			flipped1[k,:,:] = cv2.flip(image[k,:,:], 1)
		flipped2[0,:,:] = cv2.flip(label[0,:,:], 1)
	elif random_flip==4:
		flipped1 = image
		flipped2 = label
	image = flipped1
	label = flipped2

	if Verbose:
		# Show image
		plt.imshow(np.hstack( (np.squeeze(image0), 
							   np.squeeze(image), 
							   # np.squeeze(image0-image)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')

		# Show label
		plt.imshow(np.hstack( (np.squeeze(label0), 
							   np.squeeze(label), 
							   # np.squeeze(label0-label)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')
	return image, label
######################################################################################
def doSquareRotate(
	image, 
	label, 
	random_rotatedeg = np.random.choice([0, 90, 180, 270]), 
	Verbose=False):
	""" 
	Perform random rotation
	"""
	# image = np.squeeze(image)
	# label = np.squeeze(label)

	assert((image.shape[2] == label.shape[2]) and (image.shape[1] == label.shape[1]))
	shape = image.shape
	dimk, dimy, dimx = shape

	if Verbose:
		image0 = image
		label0 = label

	#random_rotatedeg = random.choice([0, 90, 180, 270])
	shiftedDistance = 0
	rot_mat = cv2.getRotationMatrix2D(                          \
		(dimx/2+randint(-shiftedDistance, shiftedDistance), 	\
		 dimy/2+randint(-shiftedDistance, shiftedDistance)),    \
		random_rotatedeg, 1.0)
	rotated1 = np.zeros(shape)
	rotated2 = np.zeros((1, dimy, dimx))
	for k in range(dimk):
		rotated1[k,:,:] = cv2.warpAffine(image[k,:,:], rot_mat, (dimx, dimy))
	rotated2[0,:,:] = cv2.warpAffine(label[0,:,:], rot_mat, (dimx, dimy))
	image = rotated1
	label = rotated2
	if Verbose:
		# Show image
		plt.imshow(np.hstack( (np.squeeze(image0), 
							   np.squeeze(image), 
							   # np.squeeze(image0-image)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')

		# Show label
		plt.imshow(np.hstack( (np.squeeze(label0), 
							   np.squeeze(label), 
							   # np.squeeze(label0-label)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')
	return image, label
######################################################################################
def doElastic(
	image, 
	label, 
	Verbose=False):
	""" 
	Perform  transformation using elastic deformation
	"""

	# Squeeze the singleton dimensional of image
	image = np.squeeze(image)
	label = np.squeeze(label)

	# 012 to 120
	# image = np.transpose(image, (1, 2, 0))
	# label = np.transpose(label, (1, 2, 0))
	if Verbose:
		print "Perform  transformation using elastic deformation"
		print "Shape of image: ", image.shape
		print "Shape of label: ", label.shape
		pass

	# Retrieve image shape
	assert((image.shape[0] == label.shape[0]) and (image.shape[1] == label.shape[1]))
	shape = image.shape
	dimx, dimy = shape
	


	# http://stackoverflow.com/questions/11379214/random-vector-plot-in-matplotlib
	# x, y, u, v= np.random.random((4,10))
	# plt.quiver(x, y, u, v)
	# plt.show()
	size = 8
	ampl = 8
	du = np.random.uniform(-ampl, ampl, size=(size, size))
	dv = np.random.uniform(-ampl, ampl, size=(size, size))

	# Done distort at boundary
	du[ 0,:] = 0
	du[-1,:] = 0
	du[:, 0] = 0
	du[:,-1] = 0
	dv[ 0,:] = 0
	dv[-1,:] = 0
	dv[:, 0] = 0
	dv[:,-1] = 0
	if Verbose:
		plt.quiver(du, dv)
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')
	# Interpolate du
	DU = cv2.resize(du, (dimx, dimx)) 
	DV = cv2.resize(dv, (dimx, dimx)) 
	
	X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
	indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
	if Verbose:
		image0 = image
		label0 = label
	image = map_coordinates(image, indices, order=1).reshape(shape)
	label = map_coordinates(label, indices, order=1).reshape(shape)
	
	# if Verbose:
		# print np.median(label)
		# print indices
	
	if Verbose:
		# Show image
		plt.imshow(np.hstack( (np.squeeze(image0), 
							   np.squeeze(image), 
							   # np.squeeze(image0-image)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')

		# Show label
		plt.imshow(np.hstack( (np.squeeze(label0), 
							   np.squeeze(label), 
							   # np.squeeze(label0-label)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')

	return image, label
######################################################################################
def addNoise(
	image, 
	label,
	noise_type="gauss",
	Verbose = False):
	"""
	Add noise to the image, not the label
	"""
	image = np.squeeze(image)
	label = np.squeeze(label)
	# Retrieve image shape
	# assert((image.shape[0] == label.shape[0]) and (image.shape[1] == label.shape[1]))
	shape = image.shape
	# dimx, dimy = shape
	
	if Verbose:
		print "Perform random noise"
		print "Shape of image: ", image.shape
		print "Shape of label: ", label.shape
		pass
	if noise_type == "gauss":
		mean = 0 # 0 0.5
		#var = 0.1
	   	#sigma = var**0.5
		gauss = np.random.normal(mean,1,shape)
		gauss = gauss.reshape(shape)
		noisy = image + gauss
	if Verbose:
		# Show image
		plt.imshow(np.hstack( (np.squeeze(noisy), 
							   np.squeeze(image), 
							   np.squeeze(image-noisy)
							  ), 
							  ), cmap = plt.get_cmap('gray'))
		plt.axis('off')
		plt.show()
		prefix = time.strftime("%Y-%m-%d:%H:%M:%S.png", time.gmtime())
		plt.savefig("tmp/"+prefix,bbox_inches='tight')

		
	return noisy, label
######################################################################################
if __name__ == '__main__': 
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)

	# Extract one image
	image = np.squeeze(train_image[0,:,:])
	label = np.squeeze(train_label[0,:,:])

	# Test elastic
	# image, label = doElastic(image, label, Verbose=True)

	# Test random_rotate
	# image, label = doSquareRotate(image, label, Verbose=True)

	# Test random flip
	# image, label = doFlip(image, label, Verbose=True)

	# Test add noise
	# image, label = addNoise(image, label, Verbose=True)