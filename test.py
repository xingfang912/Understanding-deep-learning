"""
Created on Thu Mar  9 18:08:24 2017

@author: xing
"""

from rafD_main import load_and_test

def main(mini_batch_size,net):
    load_and_test(mini_batch_size,net)    
    
    

             
if __name__ == '__main__':
   mini_batch_size = 12
   net = 'Tiny_VGG'
   main(mini_batch_size,net)
