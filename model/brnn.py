#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 11:05:30 2019

@author: sharontan
"""

#!/usr/bin/env python
# coding: utf-8

import sklearn.metrics
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense,  GRU, Bidirectional, Activation, BatchNormalization, TimeDistributed
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2

import numpy as np
import math
import os
import pandas as pd
import datetime
from pandas import read_csv,DataFrame
from numpy import log10
from datetime import timedelta
import random
import operator
import gc



# init data



scaler = StandardScaler()


class neuralNetwork():
    def myBiRNN(gru_units=20,
                dense_units=5,
                input_shape=(None,1),
                drop_out=0.2,
                patience=5):

        model = Sequential()
        reg = L1L2(l1=0.2, l2=0.2)
        model.add(Bidirectional(GRU(units=gru_units,activation='tanh',recurrent_activation='relu',recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat"))
        '''
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(dense_units,activation='relu')))
        model.add(BatchNormalization())

        model.add(Bidirectional(GRU(units=gru_units,dropout=drop_out,activation='tanh',recurrent_activation='relu',recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat")) 

        model.add(BatchNormalization())
        '''
        model.add(Dense(units=1))
        #model.add(Activation('softmax'))
        model.add(GlobalAveragePooling1D())
        print (model.summary())
       
        
        early_stopping=EarlyStopping(monitor="var_loss", patience=patience)
        #optimizer = keras.optimizers.Adamax(learning_rate=0.005)
        model.compile(optimizer='adamax',loss='mean_squared_error',metrics=['mae'])

        
        return model