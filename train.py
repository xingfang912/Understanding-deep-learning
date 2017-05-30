"""
Created on Thu Mar  9 18:08:24 2017

@author: xing
"""
from rafD_data_processing import data_processing
from rafD_main import prepare_net,training

def main(directory,mini_batch_size,image_per_direction,max_epoch,net):
    num_of_classes,train_set_x,\
    train_set_y,test_set_x,\
    test_set_y,num_train_batches,num_test_batches = data_processing(directory,mini_batch_size,image_per_direction)
    
    network, X, y, index = prepare_net(num_of_classes=num_of_classes,net=net)
    
    training(network,X,y,index,
             train_set_x,train_set_y,
             test_set_x,test_set_y,
             mini_batch_size,num_train_batches,
             num_test_batches,net,max_epoch)

             
if __name__ == '__main__':
   directory = 'The path to the directory where the cropped images are saved at.'
   mini_batch_size = 12
   image_per_direction = 12
   max_epoch = 100
   net='Tiny_VGG'
   main(directory,mini_batch_size,image_per_direction,max_epoch,net)
