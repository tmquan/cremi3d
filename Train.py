#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""
from Model 			import *
from Deploy			import *
from Augmentation	import *
from sklearn.cross_validation 		import KFold # For cross_validation
import logging
import numpy as np
import mxnet as mx

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
######################################################################################
def augment_data(X, y):
	for k in range(len(X)):
		# image  = np.transpose(X[k], (0, 1, 2))
		# label  = np.transpose(y[k], (0, 1, 2))
		image = X[k]
		label = y[k]
		# print image.shape
		# image, label = doElastic(image, label, Verbose=False)
		image, label = doSquareRotate(image, label)
		image, label = doFlip(image, label)
		image, label = addNoise(image, label)

		# image = np.expand_dims(image, axis=1)
		# label = np.expand_dims(label, axis=1)
		
		X[k] = image
		y[k] = label 
	return X, y 
######################################################################################
def get_model():
	devs = [mx.gpu(2)]
	network = symmetric_residual()
	
	model = mx.model.FeedForward(ctx=devs,
		symbol          = network,
		num_epoch       = 1,
		learning_rate	= 0.001,
        wd				= 0.000001,
        momentum		= 0.9,
		optimizer       = mx.optimizer.RMSProp(),
		initializer     = mx.init.Xavier(rnd_type="gaussian", 
						  factor_type="in", 
						  magnitude=1.23),
		)	
	return model
######################################################################################

def train():

	## Load the data 
	print "Load the data"
	X_A = np.load('sample_A_20160501.hdf_image3d_radius_15.npy')
	X_B = np.load('sample_B_20160501.hdf_image3d_radius_15.npy')
	X_C = np.load('sample_C_20160501.hdf_image3d_radius_15.npy')
	y_A = np.load('sample_A_20160501.hdf_membr3d_radius_15.npy')
	y_B = np.load('sample_B_20160501.hdf_membr3d_radius_15.npy')
	y_C = np.load('sample_C_20160501.hdf_membr3d_radius_15.npy')
	
	X  = np.concatenate((X_A, X_B, X_C), axis=0)
	y  = np.concatenate((y_A, y_B, y_C), axis=0)

	##################################################################################
	# X  = X/255.0
	y  = y/255.0
	# skimage.io.imsave('y_train.tif', y)
	##################################################################################
	# X = X[0:30,...]
	# y = y[0:30,...]
	##################################################################################
	#nb_iter 			= 1001	 #check DeepCSMRI =nb_iter
	#epochs_per_iter    = 1  #check DeepCSMRI =2
	#batch_size 		= 20 #check DeepCSMRI =20
	
	model = get_model()
	
	
	# nb_folds = 3 check DeepCSMRI
	kfolds = KFold(len(X), nb_folds)
	for iter in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(iter, nb_iter))  
		print('-'*50) 
		
		print X.shape
		print y.shape
		
		# X = np.reshape(X, 	(-1, 256, 256, 1))
		# y = np.reshape(X, 	(-1, 256, 256, 1))

		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		
		
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print('Augmenting data for training...')
			X_train, y_train = augment_data(X_train, y_train) # Data augmentation for training 
			X_valid, y_valid = augment_data(X_valid, y_valid) # Data augmentation for training 
			
			print "X_train", X_train.shape
			print "y_train", y_train.shape
			
			print "X_valid", X_valid.shape
			print "y_valid", y_valid.shape
			
			y_train  = np.concatenate((y_train, 1-y_train), axis=1)
			# # y_train  = y_train.astype(np.uint32)
			y_valid  = np.concatenate((y_valid, 1-y_valid), axis=1)
			# # y_valid  = y_valid.astype(np.uint32)

			data_train = mx.io.NDArrayIter(X_train, y_train,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, y_valid,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			model.fit(X = data_train, 
					  # eval_metric = RMSECustom(), 
					  eval_metric = mx.metric.RMSE(),
					  # eval_metric = mx.metric.Accuracy(),
					  # eval_metric = mx.metric.CustomMetric(skimage.measure.compare_psnr),
					  # eval_metric = mx.metric.MAE(),
					  eval_data = data_valid,
					  # batch_end_callback = mx.callback.Speedometer(100, 100)
					  # monitor=mon 
							)
		del X_train, X_valid, y_train, y_valid
		if iter%1==0:
			# fname = 'model_direct_%05d.tfl' %(iter)
			# model.save(fname)
			# deploy_db(model, iter)
			model.save('models/model', iter)
			print "Predicting on 30 slices"
			deploy(iter)
			# X_deploy = X[0:30,...]
			# pred  = model.predict(X_deploy, num_batch=None)
			# pred  = np.reshape(pred, (30,1,1280, 1280))
			# pred  = pred[:,:,radius:-radius,radius:-radius];
			# pred  = np.array(pred[:,0,:,:])	
			# skimage.io.imsave('./tmp/result_train_'+str(i).zfill(3)+'.tif', pred)
if __name__ == '__main__':
	import os
	# os.environ["CUDA_VISIBLE_DEVICES"]=str(np.random.choice([1, 2, 3]))
	# os.environ["CUDA_VISIBLE_DEVICES"]="1"
	train()
