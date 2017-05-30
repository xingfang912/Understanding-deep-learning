"""
Created on Thu Mar  9 15:31:13 2017

@author: xing
"""
import theano.tensor as T
import os,cv2,pickle, theano
import numpy as np

def load_data(data):
    
    return T._shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    
    
def load_label(labels):
    
    shared_y = T._shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)

    return T.cast(shared_y, 'int32')


def read_img(path, colorspace='bgr', normalize=True):
    img = cv2.imread(path)
    if colorspace == 'gray' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colorspace == 'bgr' and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if normalize:
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = img.astype(dtype=np.float32)/256.0
    return img

    
def data_processing(directory,mini_batch_size,image_per_direction,seed = 22345):
    if not os.path.isfile('testing_labels.p'):
        data = {0:{},45:{},90:{},135:{},180:{}}
        labels = []
        #read the images from the directory
        for subdir, dirs, files in os.walk(directory):
            for image in files:
                r = read_img(directory+'/'+image)
                s = image.split('_')
                if int(s[1]) not in labels:
                    labels.append(int(s[1]))
                    
                    
                if s[0][-2:]=='00':
                    try:
                        data[0][int(s[1])].append([r,int(s[1])])
                    except KeyError:
                        data[0][int(s[1])] = [[r,int(s[1])]]
                elif s[0][-2:]=='45':
                    try:
                        data[45][int(s[1])].append([r,int(s[1])])
                    except KeyError:
                        data[45][int(s[1])] = [[r,int(s[1])]]
                elif s[0][-2:]=='90':
                    try:
                        data[90][int(s[1])].append([r,int(s[1])])
                    except KeyError:
                        data[90][int(s[1])] = [[r,int(s[1])]]
                elif s[0][-2:]=='35':
                    try:
                        data[135][int(s[1])].append([r,int(s[1])])
                    except KeyError:
                        data[135][int(s[1])] = [[r,int(s[1])]]
                elif s[0][-2:]=='80':
                    try:
                        data[180][int(s[1])].append([r,int(s[1])])
                    except KeyError:
                        data[180][int(s[1])] = [[r,int(s[1])]]        
        
        pickle.dump(labels,open('ground_truth.p','wb'))
        num_of_classes = len(labels)
        randomState = np.random.RandomState(seed)
        train_set = list(); test_set = list()
        #give the real label to the list
        for k,v in data.iteritems():
            for k2, v2 in v.iteritems():
                temp = []
                for item in data[k][k2]:
                    temp.append((item[0],labels.index(item[1])))
                randomState.shuffle(temp)    
                data[k][k2] = temp
                for image_tuple in data[k][k2][:image_per_direction]:
                    test_set.append(image_tuple)
                for image_tuple in data[k][k2][image_per_direction:]:
                    train_set.append(image_tuple)
                    
        randomState.shuffle(train_set)
        randomState.shuffle(test_set)
        
        num_train_batches = len(train_set)/mini_batch_size 
        num_test_batches = len(test_set)/mini_batch_size
        
        #training_data and training_labels are two numpy arrays
        training_data = np.asarray([image_tuple[0].transpose(2,0,1).reshape(3,224,224) for image_tuple in train_set])
        training_labels = np.asarray([image_tuple[1] for image_tuple in train_set]) 
    
        testing_data = np.asarray([image_tuple[0].transpose(2,0,1).reshape(3,224,224) for image_tuple in test_set])
        testing_labels = np.asarray([image_tuple[1] for image_tuple in test_set])
        
        print 'Saving data for further use...'
        pickle.dump(training_data,open('training_data.p','wb'))
        pickle.dump(testing_data,open('testing_data.p','wb'))
        pickle.dump(training_labels,open('training_labels.p','wb'))
        pickle.dump(testing_labels,open('testing_labels.p','wb'))
        
    else:
        print 'Loading data from files...'
        training_data = pickle.load(open('training_data.p'))
        training_labels = pickle.load(open('training_labels.p'))
        testing_data = pickle.load(open('testing_data.p'))
        testing_labels = pickle.load(open('testing_labels.p'))
        
    num_train_batches = training_data.shape[0]/mini_batch_size 
    num_test_batches = testing_data.shape[0]/mini_batch_size
    num_of_classes = len(set(training_labels))
    
    
    #load data to theano tensor variables
    train_set_x = load_data(training_data)
    train_set_y = load_label(training_labels)
    
    test_set_x = load_data(testing_data)
    test_set_y = load_label(testing_labels)            
    
    return num_of_classes,train_set_x,train_set_y,test_set_x,test_set_y,num_train_batches,num_test_batches 
