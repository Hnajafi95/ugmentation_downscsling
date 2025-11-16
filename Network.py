#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file for multivariate bias correction
#author          :Fang Wang
#date            :2023/2/22
#usage           :from Network import Generator
#python_version  :3.10

# Modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

# Residual block
def res_block_gen(model, kernal_size, filters, strides, initializer):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_initializer=initializer)(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_initializer=initializer)(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = add([gen, model])
    
    return model
    
class Generator(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def generator(self):
        init = RandomNormal(stddev=0.02)
        
        gen_input = Input(shape = self.noise_shape)
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    
        gen_model = model
        
        # Using 16 Residual Blocks
        for index in range(16):
	        model = res_block_gen(model, 3, 64, 1, init)
  
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=init)(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([gen_model, model])
	    
        # After the residual blocks and the addition of the skip connection

        # Upsampling Block 1: Upsample by factor of 2
        model = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
        model = UpSampling2D(size=2)(model)
        model = PReLU(shared_axes=[1,2])(model)

        # Upsampling Block 2: Upsample by factor of 3 (New Block)
        model = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
        model = UpSampling2D(size=3)(model)
        model = PReLU(shared_axes=[1,2])(model)

        # Upsampling Block 3: Upsample by factor of 2
        model = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
        model = UpSampling2D(size=2)(model)
        model = PReLU(shared_axes=[1,2])(model)

        model = SpatialDropout2D(rate=0.1)(model, training=True)

        # Final output layer
        model = Conv2D(filters=1, kernel_size=9, strides=1, padding="same", kernel_initializer=init)(model)


	   
        generator_model = Model(inputs = gen_input, outputs = model)
        return generator_model
     
