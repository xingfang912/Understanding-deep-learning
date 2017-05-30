"""
Created on Thu Mar  9 15:37:11 2017

@author: xing
"""
from rafD_data_processing import load_data,load_label#,data_processing
import CNN_library as CNN
import DECNN_library as DECNN
import theano.tensor as T
import sys, lasagne, pickle
import numpy as np
import theano, time

def training(network,X,y,index,train_set_x,train_set_y,test_set_x,test_set_y,mini_batch_size,num_train_batches,num_test_batches,net,max_epoch):
    prediction = lasagne.layers.get_output(network['prob'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network.values(), trainable=True)

    
    updates = lasagne.updates.sgd(
            loss, params, learning_rate=0.01)
            
    train_fn = theano.function([index], loss, updates=updates, givens={X:train_set_x[index*mini_batch_size:(index+1) * mini_batch_size], 
                                                                       y:train_set_y[index*mini_batch_size:(index+1)*mini_batch_size]}, on_unused_input='ignore')
    
    # Launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    best_test_acc = -1
    for epoch in range(max_epoch):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(num_train_batches):
            if (i+1) % 20 == 0:
                print 'training at {} epoch and {} iteration'.format(epoch+1,i+1)
                                                                  
            train_err += train_fn(i)
            train_batches += 1
            
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print 'Testing...'
        current_test_acc,predicts = testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches)
	print "  test accuracy:\t\t{:.2f} %".format(current_test_acc)
        if current_test_acc > best_test_acc:
            print 'Saving the network...'
            np.savez(net+'.npz', *lasagne.layers.get_all_param_values(network.values()))
            print 'Network saved!'
            best_test_acc = current_test_acc

    save_miss_classified(predicts,net)	

    print 'Training complete!'                                                                


def testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches):
    
    test_prediction = lasagne.layers.get_output(network['prob'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,y)
    test_loss = test_loss.mean()
    prediction_results = T.argmax(test_prediction, axis=1)
    test_accuracy = T.eq(prediction_results, y)

    test_fn = theano.function([index], [prediction_results, test_accuracy], givens={X:test_set_x[index*mini_batch_size:(index+1)*mini_batch_size],
                                                                      y:test_set_y[index*mini_batch_size:(index+1)*mini_batch_size]}, on_unused_input='ignore')    
    
    predicts = [];correctly_classified_cases = 0
    test_acc=0;test_batches=0
    for i in range(num_test_batches):
        pred,acc = test_fn(i)
        correctly_classified_cases += list(acc).count(1)
        test_batches += 1
        predicts += list(pred)
            
        
    current_test_acc = correctly_classified_cases/float(num_test_batches*mini_batch_size) * 100
	
    return current_test_acc,predicts

def load_and_test(mini_batch_size,net):
    print 'Loading testing data...'
    test_set_x = load_data(pickle.load(open('testing_data.p')))
    labels = pickle.load(open('testing_labels.p'))
    num_test_batches = labels.shape[0]/mini_batch_size
    test_set_y = load_label(labels)
    num_of_classes = len(set(labels))
    
    print 'Testing...'
    if net == 'Tiny_VGG':
        with np.load(net+'.npz') as f:
            trained_params = [f['arr_%d' % i] for i in range(len(f.files))]
        network,X,y,index = prepare_net(num_of_classes=num_of_classes,net=net)
        lasagne.layers.set_all_param_values(network.values(),trained_params)
        test_acc,predicts = testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches)
        print 'Testing accuracy: %.2f%%'%(test_acc)
    elif net == 'Fully_Conv_4':
        with np.load(net+'.npz') as f:
            trained_params = [f['arr_%d' % i] for i in range(len(f.files))]
        network,X,y,index = prepare_net(num_of_classes=num_of_classes,net=net)
        lasagne.layers.set_all_param_values(network.values(),trained_params)
        test_acc,predicts = testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches)
        print 'Testing accuracy: %.2f%%'%(test_acc)
    elif net == 'Fully_Conv_2':
        with np.load(net+'.npz') as f:
            trained_params = [f['arr_%d' % i] for i in range(len(f.files))]
        network,X,y,index = prepare_net(num_of_classes=num_of_classes,net=net)
        lasagne.layers.set_all_param_values(network.values(),trained_params)
        test_acc,predicts = testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches)
        print 'Testing accuracy: %.2f%%'%(test_acc)
    elif net == 'VGG_16':
        with np.load(net+'.npz') as f:
            trained_params = [f['arr_%d' % i] for i in range(len(f.files))]
        network,X,y,index = prepare_net(num_of_classes=num_of_classes,net=net)
        lasagne.layers.set_all_param_values(network.values(),trained_params)
        test_acc,predicts = testing(network,X,y,index,test_set_x,test_set_y,mini_batch_size,num_test_batches)
        print 'Testing accuracy: %.2f%%'%(test_acc)
    else:
        sys.exit('No such network.')
    


def save_miss_classified(predicts,net):
    testing_labels = pickle.load(open( "testing_labels.p", "rb" ))
    miss_classified_cases = [(i,predicts[i],testing_labels[i]) for i in range(len(predicts)) if predicts[i] != testing_labels[i]]
    pickle.dump(miss_classified_cases,open('miss_classified'+'_'+net+'.p','wb'))    
    
def prepare_net(num_of_classes,net):
        
    # Prepare Theano variables for inputs and targets
    X = T.tensor4('x')
    y = T.ivector('y')
    index = T.lscalar()
    
    if net == 'Tiny_VGG':
        network = CNN.Tiny_VGG(num_of_classes=num_of_classes,input_var=X)
    elif net == 'Fully_Conv_4':
        network = CNN.Fully_Conv_4(num_of_classes=num_of_classes,input_var=X)
    elif net == 'Fully_Conv_6':
        network = CNN.Fully_Conv_6(num_of_classes=num_of_classes,input_var=X)
    elif net == 'VGG_16':
        network = CNN.VGG_16(num_of_classes=num_of_classes,input_var=X)
    else:		
        sys.exit('No such network.')
        
    return network, X, y, index 
    

def prepare_deconv_net(params,net):
        
    # Prepare Theano variables for inputs and targets
    X = T.tensor4('x')
    y = T.ivector('y')
    index = T.lscalar()
    
#    X = X.reshape((mini_batch_size,3,224,224))
    
    if net == 'Tiny_VGG':
        network = DECNN.Deconv_Tiny_VGG(params=params,input_var=X)
    elif net == 'Fully_Conv_4':
        network = DECNN.Deconv_Fully_Conv_4(params=params,input_var=X)
    else:		
        sys.exit('No such network.')
        
    return network, X, y, index   
