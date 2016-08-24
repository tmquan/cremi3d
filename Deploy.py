from Utility import *
from Model 	 import *
from Utility import *
from Config  import * # radius
from scipy.ndimage.interpolation import *
	
def deploy(iter):
	# X = np.load('X_train.npy')

	samples = np.load('sample_A_20160501.hdf_image_radius_15.npy')
	X = samples[0:30,...]

	print "X.shape", X.shape
	
	
	print "X.shape", X.shape
	shape = X.shape
	
	X_deploy = X
	
	print "X_deploy.shape", X_deploy.shape
	# print "y_deploy.shape", y_deploy.shape
	# Load model
	# iter = 100
	model 	= mx.model.FeedForward.load('models/model', iter, ctx=mx.gpu(3))
	# Perform prediction
	# batch_size = 1
	print('Predicting on data...')
	
	
	# Original maps
	X_deploy = X
	pred_recon0  = model.predict(X_deploy, num_batch=None)
	print "pred_recon0.shape", pred_recon0.shape
	
	pred_recon0  = pred_recon0[:,0,:,:]
	pred_recon0  = np.expand_dims(pred_recon0, axis=1)




	# Rotate 90
	X_deploy = scipy.ndimage.interpolation.rotate(X, angle=90, axes=(2,3))
	pred_recon1  = model.predict(X_deploy, num_batch=None)
	print "pred_recon1.shape", pred_recon1.shape
	pred_recon1  = scipy.ndimage.interpolation.rotate(pred_recon1, angle=-90, axes=(2,3))
	pred_recon1  = pred_recon1[:,0,:,:]
	pred_recon1  = np.expand_dims(pred_recon1, axis=1)

	# Rotate 180
	X_deploy = scipy.ndimage.interpolation.rotate(X, angle=180, axes=(2,3))
	pred_recon2  = model.predict(X_deploy, num_batch=None)
	print "pred_recon2.shape", pred_recon2.shape
	pred_recon2  = scipy.ndimage.interpolation.rotate(pred_recon2, angle=-180, axes=(2,3))
	pred_recon2  = pred_recon2[:,0,:,:]
	pred_recon2  = np.expand_dims(pred_recon2, axis=1)

	# Rotate 270
	X_deploy = scipy.ndimage.interpolation.rotate(X, angle=270, axes=(2,3))
	pred_recon3  = model.predict(X_deploy, num_batch=None)
	print "pred_recon3.shape", pred_recon3.shape
	pred_recon3  = scipy.ndimage.interpolation.rotate(pred_recon3, angle=-270, axes=(2,3))
	pred_recon3  = pred_recon3[:,0,:,:]
	pred_recon3  = np.expand_dims(pred_recon3, axis=1)
		
	# Flip left right
	X_deploy = X[...,::1,::-1]
	pred_recon4  = model.predict(X_deploy, num_batch=None)
	print "pred_recon4.shape", pred_recon4.shape
	pred_recon4  = pred_recon4[...,::1,::-1]
	pred_recon4  = pred_recon4[:,0,:,:]
	pred_recon4  = np.expand_dims(pred_recon4, axis=1)

	# Rotate 90 then Flip left right
	X_deploy = X[...,::1,::-1]
	X_deploy = scipy.ndimage.interpolation.rotate(X_deploy, angle=90, axes=(2,3))
	pred_recon5  = model.predict(X_deploy, num_batch=None)
	print "pred_recon5.shape", pred_recon5.shape
	pred_recon5  = scipy.ndimage.interpolation.rotate(pred_recon5, angle=-90, axes=(2,3))
	pred_recon5  = pred_recon5[...,::1,::-1]
	pred_recon5  = pred_recon5[:,0,:,:]
	pred_recon5  = np.expand_dims(pred_recon5, axis=1)
	
	# Rotate 180 then Flip left right
	X_deploy = X[...,::1,::-1]
	X_deploy = scipy.ndimage.interpolation.rotate(X_deploy, angle=180, axes=(2,3))
	pred_recon6  = model.predict(X_deploy, num_batch=None)
	print "pred_recon6.shape", pred_recon6.shape
	pred_recon6  = scipy.ndimage.interpolation.rotate(pred_recon6, angle=-180, axes=(2,3))
	pred_recon6  = pred_recon6[...,::1,::-1]
	pred_recon6  = pred_recon6[:,0,:,:]
	pred_recon6  = np.expand_dims(pred_recon6, axis=1)	
	
	# Rotate 270 then Flip left right
	X_deploy = X[...,::1,::-1]
	X_deploy = scipy.ndimage.interpolation.rotate(X_deploy, angle=270, axes=(2,3))
	pred_recon7  = model.predict(X_deploy, num_batch=None)
	print "pred_recon7.shape", pred_recon7.shape
	pred_recon7  = scipy.ndimage.interpolation.rotate(pred_recon7, angle=-270, axes=(2,3))
	pred_recon7  = pred_recon7[...,::1,::-1]
	pred_recon7  = pred_recon7[:,0,:,:]
	pred_recon7  = np.expand_dims(pred_recon7, axis=1)
	
	# Join the prediction
	pred_recon = np.concatenate((pred_recon0, pred_recon1, pred_recon2, pred_recon3, pred_recon4, pred_recon5, pred_recon6, pred_recon7), axis=1);	
	# pred_recon = np.concatenate((pred_recon0, pred_recon1), axis=1);	
	# pred_recon = np.mean(pred_recon, axis=1);
	pred_recon_avg = np.mean(pred_recon, axis=1);
	pred_recon_med = np.median(pred_recon, axis=1);
	# pred_recon = np.amin(pred_recon, axis=1);
	print "pred_recon.shape", pred_recon.shape
	pred_recon_avg = pred_recon_avg[...,radius:-radius, radius:-radius]
	pred_recon_med = pred_recon_med[...,radius:-radius, radius:-radius]

	prefix = time.strftime("%Y-%m-%d:%H:%M:%S_", time.gmtime())
	print prefix
	skimage.io.imsave("result/"+prefix+"leaky_avg_"+str(iter).zfill(5)+".tif", pred_recon_avg)
	skimage.io.imsave("result/"+prefix+"leaky_med_"+str(iter).zfill(5)+".tif", pred_recon_med)

####################################################################################################
if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="3"
	iter = 0
	deploy(iter)
