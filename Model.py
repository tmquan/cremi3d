# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 20num_filter_scale1

@author: tmquan
"""

from Utility import *
import mxnet as mx

	    	   
def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
	if conv_type == 0:
		conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
		# bn = mx.symbol.BatchNorm(data=conv)
		act = mx.symbol.Activation(data = conv, act_type=act_type)
		# act = mx.symbol.Dropout(data = act, p=0.25)
		return act
	elif conv_type == 1:
		conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
		bn = mx.symbol.BatchNorm(data=conv)
		return bn



# def convolution_factory(data, num_filter, kernel, stride, pad):	
	# conv 	= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
    # bn 		= mx.symbol.BatchNorm(data=conv)
    # act 	= mx.symbol.Activation(data = bn, act_type='relu')
	
def residual_factory(data, num_filter, kernel, stride, pad):	
	identity_data 	= data
	
	conv1 			= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	# bn1 			= mx.symbol.BatchNorm(data = conv1)
	# act1 			= mx.symbol.Activation(data = conv1, act_type='relu')
	
	conv2 			= mx.symbol.Convolution(data = conv1, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	# bn2 			= mx.symbol.BatchNorm(data = conv2)
	act2 			= mx.symbol.Activation(data = conv2, act_type='relu')
	
	conv3 			= mx.symbol.Convolution(data = act2, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	# new_data 		= conv3
	new_data 		= conv3 + identity_data
	# new_data 		= conv3 - identity_data
	# new_data 		= conv3 * identity_data
	# new_data 		= mx.symbol.Concat(*[conv3, identity_data])
	# new_data 		= mx.symbol.Dropout(data = new_data, p=0.25)
	# bn3 			= mx.symbol.BatchNorm(data = new_data)
	act3 			= mx.symbol.Activation(data = new_data, act_type='relu')
	# act3 			= act3 + identity_data
	return act3

num_filter_scale0 = 32 #64  # 32 	#
num_filter_scale1 = 48 # 64 # 32 	#
num_filter_scale2 = 64 # 128 # 64 	#
num_filter_scale3 = 80 #256 # 96 	#
num_filter_scale4 = 96 #384 # 128 #
num_filter_scale5 = 128 #512 # 256 #

def symmetric_residual():
	
	
	data = mx.symbol.Variable('data')
	data = data/255
	
	# Before down
	conv0a = conv_factory(
		data		= 	data, 
		num_filter	=	num_filter_scale0, 
		kernel		=	(31,31), 
		stride		=	(1,1), 
		pad			= 	(15,15), 
		act_type = 'relu')
	
	
	####################################################################
	scale = 2
	workspace_default = 1024	
	####################################################################
	# Residual with concat is here
	conv1a = conv_factory(
		data		= 	conv0a, 
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res1a = residual_factory(
		data 		= 	conv1a,
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	
	down1a = mx.symbol.Pooling(
		data		= res1a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	down1a = mx.symbol.Dropout(down1a)
	####################################################################	
	conv2a = conv_factory(
		data		= 	down1a, 
		num_filter	=	num_filter_scale2, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res2a = residual_factory(
		data 		= 	conv2a,
		num_filter	=	num_filter_scale2, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down2a = mx.symbol.Pooling(
		data		= res2a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	down2a = mx.symbol.Dropout(down2a)
	####################################################################
	conv3a = conv_factory(
		data		= 	down2a, 
		num_filter	=	num_filter_scale3, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res3a = residual_factory(
		data 		= 	conv3a,
		num_filter	=	num_filter_scale3, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down3a = mx.symbol.Pooling(
		data		= res3a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	down3a = mx.symbol.Dropout(down3a)
	####################################################################
	conv4a = conv_factory(
		data		= 	down3a, 
		num_filter	=	num_filter_scale4, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res4a = residual_factory(
		data 		= 	conv4a,
		num_filter	=	num_filter_scale4, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down4a = mx.symbol.Pooling(
		data		= res4a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	down4a = mx.symbol.Dropout(down4a)
	####################################################################
	conv_mid = conv_factory(
		data		= 	down4a, 
		num_filter	=	num_filter_scale5, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up4b = mx.symbol.Deconvolution(
		data		=	conv_mid, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	num_filter_scale4, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat4b = mx.symbol.Concat(*[up4b, res4a])
	# ccat4b = mx.symbol.Group([up4b, res4a])
	# ccat4b = up4b
	# ccat4b = up4b - res4a
	ccat4b = up4b + res4a
	# ccat4b = up4b * res4a
	ccat4b = conv_factory(
		data		= 	ccat4b, 
		num_filter	=	num_filter_scale4, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res4b = residual_factory(
		data 		= 	ccat4b,
		num_filter	=	num_filter_scale4, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv4b = conv_factory(
		data		= 	res4b, 
		num_filter	=	num_filter_scale3, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up3b = mx.symbol.Deconvolution(
		data		=	conv4b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	num_filter_scale3, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat3b = mx.symbol.Concat(*[up3b, res3a])
	# ccat3b = mx.symbol.Group([up3b, res3a])
	# ccat3b = up3b
	# ccat3b = up3b - res3a
	ccat3b = up3b + res3a
	# ccat3b = up3b * res3a
	ccat3b = conv_factory(
		data		= 	ccat3b, 
		num_filter	=	num_filter_scale3, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res3b = residual_factory(
		data 		= 	ccat3b,
		num_filter	=	num_filter_scale3, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv3b = conv_factory(
		data		= 	res3b, 
		num_filter	=	num_filter_scale2, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up2b = mx.symbol.Deconvolution(
		data		=	conv3b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	num_filter_scale2, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat2b = mx.symbol.Concat(*[up2b, res2a])
	# ccat2b = mx.symbol.Group([up2b, res2a])
	# ccat2b = up2b
	ccat2b = up2b + res2a
	# ccat2b = up2b - res2a
	# ccat2b = up2b * res2a
	ccat2b = conv_factory(
		data		= 	ccat2b, 
		num_filter	=	num_filter_scale2, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res2b = residual_factory(
		data 		= 	ccat2b,
		num_filter	=	num_filter_scale2, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv2b = conv_factory(
		data		= 	res2b, 
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up1b = mx.symbol.Deconvolution(
		data		=	conv2b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	num_filter_scale1, 
		no_bias	=	True, 
		workspace	=	workspace_default)
	
	# ccat1b = mx.symbol.Concat(*[up1b, res1a])
	# ccat1b = mx.symbol.Group([up1b, res1a])
	# ccat1b = up1b
	# ccat1b = up1b - res1a
	ccat1b = up1b + res1a
	# ccat1b = up1b * res1a
	ccat1b = conv_factory(
		data		= 	ccat1b, 
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res1b = residual_factory(
		data 		= 	ccat1b,
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv1b = conv_factory(
		data		= 	res1b, 
		num_filter	=	num_filter_scale1, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	out = conv1b
	numclass = 2
	# After up
	# Inception unit for segmentation
	conv0b = conv_factory(
		data		= 	out, 
		num_filter	=	numclass, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	
	# bn = mx.symbol.BatchNorm(data=conv0b)
	# act = mx.symbol.Activation(data = bn, act_type='relu')
	# ft   = mx.symbol.Flatten(data=conv0b)
	sm   = mx.symbol.LogisticRegressionOutput(data=conv0b, name="softmax")
	# sm   = mx.symbol.SoftmaxOutput(data=conv0b, name="softmax")
	return sm
	
if __name__ == '__main__':
	# Draw the net

	data 	= mx.symbol.Variable('data')
	network = symmetric_residual()
	dot = mx.viz.plot_network(network,
		None,
		shape={"data" : (375, 3, 1280, 1280)}
		) 
	dot.graph_attr['rankdir'] = 'RL'
	
	
	
	
	