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
from lasagne.nonlinearities import softmax

def VGG_16(num_of_classes,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

def Tiny_VGG(num_of_classes,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 32, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 32, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 32, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 32, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    net['fc4'] = DenseLayer(net['pool3'], num_units=1024)
    net['fc4_dropout'] = DropoutLayer(net['fc4'], p=0)
    net['fc5'] = DenseLayer(net['fc4_dropout'], num_units=512)
    net['fc5_dropout'] = DropoutLayer(net['fc5'], p=0)
    net['fc6'] = DenseLayer(net['fc5_dropout'], num_units=256)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)
    net['fc7'] = DenseLayer(
        net['fc6_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc7'], softmax)

    return net 
    
def Fully_Conv_4(num_of_classes,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 4, (2,2), pad=0, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 4, pad=0, flip_filters=False)
    net['conv2_1'] = ConvLayer(
        net['conv1_2'], 128, 6, (2,2), pad=0, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 256, 6, (2,2), pad=0, flip_filters=False)
    net['fc3'] = DenseLayer(net['conv2_2'], num_units=1024)
    net['fc3_dropout'] = DropoutLayer(net['fc3'], p=0.5)
    net['fc4'] = DenseLayer(net['fc3_dropout'], num_units=512)
    net['fc4_dropout'] = DropoutLayer(net['fc4'], p=0.5)
    net['fc5'] = DenseLayer(
        net['fc4_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc5'], softmax)

    return net
    
    
def Fully_Conv_6(num_of_classes,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1'] = ConvLayer(
        net['input'], 32, 4,(2,2), pad=0, flip_filters=False)
    net['conv2'] = ConvLayer(
        net['conv1'], 64, 4, pad=0, flip_filters=False)
    net['conv3'] = ConvLayer(
        net['conv2'], 64, 4, (2,2), pad=0, flip_filters=False)
    net['conv4'] = ConvLayer(
        net['conv3'], 128, 3, (2,2), pad=0, flip_filters=False)
    net['conv5'] = ConvLayer(
        net['conv4'], 128, 4, (2,2), pad=0, flip_filters=False)
    net['conv6'] = ConvLayer(
        net['conv5'], 256, 4, (2,2), pad=0, flip_filters=False)    
    net['fc3'] = DenseLayer(net['conv6'], num_units=1024)
    net['fc3_dropout'] = DropoutLayer(net['fc3'], p=0)
    net['fc4'] = DenseLayer(net['fc3_dropout'], num_units=512)
    net['fc4_dropout'] = DropoutLayer(net['fc4'], p=0)
    net['fc5'] = DenseLayer(net['fc4_dropout'], num_units=256)
    net['fc5_dropout'] = DropoutLayer(net['fc5'], p=0)
    net['fc6'] = DenseLayer(
        net['fc5_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc6'], softmax)

    return net

def Fully_Conv_2(num_of_classes,input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['fc3'] = DenseLayer(net['pool1'], num_units=512)
    net['fc3_dropout'] = DropoutLayer(net['fc3'], p=0)
    net['fc4'] = DenseLayer(net['fc3_dropout'], num_units=512)
    net['fc4_dropout'] = DropoutLayer(net['fc4'], p=0)
    net['fc5'] = DenseLayer(
        net['fc4_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc5'], softmax)

    return net
