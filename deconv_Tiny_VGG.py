"""
Created on Fri Mar 10 16:06:35 2017

@author: xing
"""

from rafD_main import prepare_deconv_net, prepare_net
import pickle, cv2
import numpy as np
import lasagne, theano
import matplotlib.pyplot as plt
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import InputLayer, InverseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
import theano.tensor as T
from PIL import ImageEnhance, Image


def deconv_Tiny(params,input_var=None,input_var2=None):
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
    
    net2 = {}
    net2['input'] = InputLayer(shape=(None, 32, 28, 28),input_var=input_var2)
    #deconvolution starts here
    net2['unpool3'] = InverseLayer(net2['input'],net['pool3'])
    
    net2['deconv3_2'] = DeconvLayer(net2['unpool3'],num_filters=32,
                                    filter_size=net['conv3_2'].filter_size, stride=net['conv3_2'].stride,
                                    crop=net['conv3_2'].pad,
                                    W=params[10], b=params[9], flip_filters=True)
    
    net2['deconv3_1'] = DeconvLayer(net2['deconv3_2'],num_filters=32,
                                    filter_size=net['conv3_1'].filter_size, stride=net['conv3_1'].stride,
                                    crop=net['conv3_1'].pad,
                                    W=params[8], b=params[7], flip_filters=True)
                                    
    net2['unpool2'] = InverseLayer(net2['deconv3_1'],net['pool2'])                                
                                    
    net2['deconv2_2'] = DeconvLayer(net2['unpool2'],num_filters=32,
                                    filter_size=net['conv2_2'].filter_size, stride=net['conv2_2'].stride,
                                    crop=net['conv2_2'].pad,
                                    W=params[6], b=params[5], flip_filters=True)
                                                                   
    net2['deconv2_1'] = DeconvLayer(net2['deconv2_2'],num_filters=32,
                                    filter_size=network['conv2_1'].filter_size, stride=network['conv2_1'].stride,
                                    crop=network['conv2_1'].pad,
                                    W=params[4], b=params[3], flip_filters=True)
                                    
    net2['unpool1'] = InverseLayer(net2['deconv2_1'],net['pool1'])

    net2['deconv1_2'] = DeconvLayer(net2['unpool1'],num_filters=32,
                                    filter_size=net['conv1_2'].filter_size, stride=net['conv1_2'].stride,
                                    crop=net['conv1_2'].pad,
                                    W=params[2], b=params[1], flip_filters=True)
                                                                   
    net2['deconv1_1'] = DeconvLayer(net2['deconv1_2'],num_filters=3,
                                    filter_size=net['conv1_1'].filter_size, stride=net['conv1_1'].stride,
                                    crop=net['conv1_1'].pad,
                                    W=params[0], flip_filters=True)
    
    return net2

if __name__=='__main__':
    print 'Loading testing data...'
    test_data = pickle.load(open('testing_data.p'))
    labels = pickle.load(open('testing_labels.p'))

    net = 'Tiny_VGG'
    
    print 'Loading the trained parameters...'
    with np.load(net+'.npz') as f:
        trained_params = [f['arr_%d' % i] for i in range(len(f.files))]
        
    network,X,y,index = prepare_deconv_net(params=trained_params,net=net)

    network3,X3,y3,index3 = prepare_net(num_of_classes=len(set(labels)),net=net)        
    lasagne.layers.set_all_param_values(network3.values(),trained_params)    
    
    miss_cases = pickle.load(open('miss_classified'+'_'+net+'.p','rb'))
    
    ground_truth = pickle.load(open('ground_truth.p'))
    
    directory = r'/home/xing/Documents/facial/RafD-facial-recognition/Tiny_VGG results/'
   
    average_neurons = []
    for p in range(len(miss_cases)):
        print p
        
        data = test_data[miss_cases[p][0]]    
        data = data.reshape((1,3,224,224))
        
        
        
    
        deconv_output = lasagne.layers.get_output(network['deconv1_1'])
                                                                          
        deconv_fn = theano.function([], deconv_output, 
                                    givens={X:data.astype(dtype=theano.config.floatX)},on_unused_input='ignore')
        
                
        output = deconv_fn()
        o = output[0].copy()
        o = np.rollaxis(o, 0, 3)
        gray_o = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    
        infor = directory+str(miss_cases[p][0])+"_"+str(ground_truth[miss_cases[p][1]])+"_"+str(ground_truth[miss_cases[p][2]])
        
        k = 1
        plt.imsave(infor+"_"+str(k)+".png",test_data[miss_cases[p][0]][0], cmap=plt.cm.gray)
    
        
    
        test_prediction = lasagne.layers.get_output(network3['prob'], deterministic=True)
        
        pool3 = lasagne.layers.get_output(network3['pool3'], deterministic=True)
        fc4_out = lasagne.layers.get_output(network3['fc4'], deterministic=True)
        fc5_out = lasagne.layers.get_output(network3['fc5'], deterministic=True)
        fc6_out = lasagne.layers.get_output(network3['fc6'], deterministic=True)
        test = lasagne.layers.get_output(network3['fc7'], deterministic=True)
        params = lasagne.layers.get_all_param_values(network3['fc7'])
        prediction_results = T.argmax(test_prediction, axis=1)
        test_fn = theano.function([], [test_prediction,prediction_results,fc6_out,fc5_out,fc4_out,pool3], 
                                  givens={X3:data.astype(dtype=theano.config.floatX)}, on_unused_input='ignore')
        prob_distribution, predicted, fc6_out, fc5_out, fc4_out, pool3 = test_fn()
        

        ######################
        #Backtracking starts#
        ####################   
        missed_as = miss_cases[p][1]
        l = list(np.multiply(fc6_out,params[-2][:,missed_as])[0])
        neurons_fc6 = [l.index(x) for x in l if x >0]
        
        neurons_fc5 = []
        for n in neurons_fc6:
            l = list(np.multiply(fc5_out,params[-4][:,n])[0])
            neurons_fc5 += [l.index(x) for x in l if x >0]
        neurons_fc5 = list(set(neurons_fc5))
    
        neurons_fc4 = []
        for n in neurons_fc5:
            l = list(np.multiply(fc4_out,params[-6][:,n])[0])
            neurons_fc4 += [l.index(x) for x in l if x >0]
        neurons_fc4 = list(set(neurons_fc4))
        
        neurons_pool3 = [];positive_activations = []
        pool3_flatten = pool3.flatten()
        for n in neurons_fc4:
            l = list(np.multiply(pool3_flatten,params[-8][:,n]))
            positive_activations += [x for x in l if x >0.05]
        cutoff = np.percentile(positive_activations,90)
#        cutoff = 0
        print cutoff, len(neurons_fc4)
        for n in neurons_fc4:
            l = list(np.multiply(pool3_flatten,params[-8][:,n]))
            neurons_pool3 += [l.index(x) for x in l if x >cutoff]    
        neurons_pool3 = list(set(neurons_pool3))
        average_neurons.append(len(neurons_pool3))
        
        ######################
        #Backtracking ends ##
        #################### 
        
        print 'flatten...'
        #suppressing the activations
        for x in neurons_pool3:
            pool3_flatten[x] = 0   
    
        
    
        data2 = pool3_flatten.reshape(1,32,28,28)
        
        for i in range(32):
            k+=1
            b = 255-pool3[0][i]
            b = Image.fromarray(b)
            b = b.convert('L')
            e = ImageEnhance.Brightness(b)
            b = e.enhance(0.7)
            e = ImageEnhance.Contrast(b)
            b = e.enhance(40)
            b.save(infor+"_"+str(k)+".png",format='png')
        
        for i in range(32):
            k+=1
            b = 255-data2[0][i]
            b = Image.fromarray(b)
            b = b.convert('L')
            e = ImageEnhance.Brightness(b)
            b = e.enhance(0.7)
            e = ImageEnhance.Contrast(b)
            b = e.enhance(40)
            b.save(infor+"_"+str(k)+".png",format='png')
        
        X2 = T.tensor4('x2')    
        
        # Feature tracking and visualizations                            
        network2 = deconv_Tiny(trained_params,X,X2)
        deconv_output = lasagne.layers.get_output(network2['deconv1_1'])
        unpool3 = lasagne.layers.get_output(network2['unpool2'])                                                                  
        deconv_fn = theano.function([], [deconv_output,unpool3], 
                                    givens={X:data.astype(dtype=theano.config.floatX),
                                            X2:data2.astype(dtype=theano.config.floatX)},on_unused_input='ignore')                            
        
                
        output2,unpool3 = deconv_fn()
        o2 = output2[0].copy()
        o2 = np.rollaxis(o2, 0, 3)
        gray_o2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
    
   
        od = abs(gray_o2-gray_o)

        plt.imsave(infor+"_"+str(k+1)+".png",255-gray_o.astype(np.uint8),cmap=plt.cm.gray)
        plt.imsave(infor+"_"+str(k+2)+".png",255-gray_o2.astype(np.uint8),cmap=plt.cm.gray)
        b = 255-38*od.astype(np.uint8)
        b = Image.fromarray(b)
        b = b.convert('L')
        e = ImageEnhance.Brightness(b)
        b = e.enhance(0.75)
        e = ImageEnhance.Contrast(b)
        b = e.enhance(2.5)
        b.save(infor+"_"+str(k+3)+".png",format='png')
    print "Average activated neurons: "+str(sum(average_neurons)/float(len(average_neurons)))
