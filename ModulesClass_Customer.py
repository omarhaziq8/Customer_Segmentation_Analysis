# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:54:34 2022

@author: pc
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout



class ModelCreation():
    def __init__(self):
        pass
    
    def simple_bn_layer(self,x_train,num_node=32,drop_rate=0.2,
                          output_node=2):
        model = Sequential() # to create container 
        model.add(Input(shape=(np.shape(x_train)[1:])))
        model.add(Dense(num_node,activation='relu',name='HL1'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node,activation='relu',name='HL2'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='relu',name='Output'))
        model.summary()
        
        return model


class Model_Evaluation():
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'],label='Training loss')
        plt.plot(hist.history['val_loss'],label='Validation loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'],label='Training acc')
        plt.plot(hist.history['val_acc'],label='Validation acc')
        plt.legend()
        plt.show()























