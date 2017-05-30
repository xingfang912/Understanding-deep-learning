"""
Created on Thu Mar  9 15:48:42 2017

@author: xing
"""

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import InverseLayer

def Deconv_Tiny_VGG(params,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1, W=params[0], b=params[1], flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1, W=params[2], b=params[3], flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 32, 3, pad=1, W=params[4], b=params[5], flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 32, 3, pad=1, W=params[6], b=params[7], flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 32, 3, pad=1, W=params[8], b=params[9], flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 32, 3, pad=1, W=params[10], b=params[11], flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    #deconvolution starts here
    net['unpool3'] = InverseLayer(net['pool3'],net['pool3'])
    
    net['deconv3_2'] = DeconvLayer(net['unpool3'],num_filters=32,
                                    filter_size=net['conv3_2'].filter_size, stride=net['conv3_2'].stride,
                                    crop=net['conv3_2'].pad,
                                    W=params[10], b=params[9], flip_filters=True)
    
    net['deconv3_1'] = DeconvLayer(net['deconv3_2'],num_filters=32,
                                    filter_size=net['conv3_1'].filter_size, stride=net['conv3_1'].stride,
                                    crop=net['conv3_1'].pad, 
                                    W=params[8], b=params[7], flip_filters=True)
                                    
    net['unpool2'] = InverseLayer(net['deconv3_1'],net['pool2'])                                
                                    
    net['deconv2_2'] = DeconvLayer(net['unpool2'],num_filters=32,
                                    filter_size=net['conv2_2'].filter_size, stride=net['conv2_2'].stride,
                                    crop=net['conv2_2'].pad, 
                                    W=params[6], b=params[5], flip_filters=True)
                                                                   
    net['deconv2_1'] = DeconvLayer(net['deconv2_2'],num_filters=32,
                                    filter_size=net['conv2_1'].filter_size, stride=net['conv2_1'].stride,
                                    crop=net['conv2_1'].pad, 
                                    W=params[4], b=params[3], flip_filters=True)
                                    
    net['unpool1'] = InverseLayer(net['deconv2_1'],net['pool1'])

    net['deconv1_2'] = DeconvLayer(net['unpool1'],num_filters=32,
                                    filter_size=net['conv1_2'].filter_size, stride=net['conv1_2'].stride,
                                    crop=net['conv1_2'].pad, 
                                    W=params[2], b=params[1], flip_filters=True)
                                                                   
    net['deconv1_1'] = DeconvLayer(net['deconv1_2'],num_filters=3,
                                    filter_size=net['conv1_1'].filter_size, stride=net['conv1_1'].stride,
                                    crop=net['conv1_1'].pad, 
                                    W=params[0], flip_filters=True)
    
    return net
