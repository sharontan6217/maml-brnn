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
from keras.layers import Dense,  GRU, Bidirectional
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

    def myBiRNN(x_train,
                y_train,
                x_test,
                y_test,
                gru_units=None,
                dense_units=None,
                input_shape=None,
                batch_size=None,
                epochs=None,
                drop_out=None,
                patience=None):

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
        model.add(GlobalAveragePooling1D())
        print (model.summary())
        
        early_stopping=EarlyStopping(monitor="var_loss", patience=patience)
        optimizer = keras.optimizers.Adamax(learning_rate=0.005)
        model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mae'])
        
        history_callback=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,\
                                   verbose=2, validation_data=[x_test,y_test],shuffle = True)
        
        return model, history_callback
        
    def train(x_train,y_train,x_test,y_test,x_sample,y_sample):

        batch_size=32
        epochs=3000
        drop_out=0.1
        patience=5
        gru_units=20
        dense_units=5
        input_shape=(None,1)
        
        #data load
        print("#data load:")
        print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test))
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        x_train=np.reshape(x_train,(len(x_train),1,1))
        x_test=np.reshape(x_test,(len(x_test),1,1))
        #train
        model, loss_history=neuralNetwork.myBiRNN(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,gru_units=gru_units,\
                                                  input_shape=input_shape,\
                                                batch_size=batch_size,epochs=epochs)
        trainPredict=model.predict(x_train)
        trainScore=math.sqrt(mean_squared_error(y_train,trainPredict)) 
        print('Train Score: %.5f RMSE' % (trainScore))
        
        testPredict=model.predict(x_test)
        testScore=math.sqrt(mean_squared_error(y_test,testPredict)) 
        print('Test Score: %.5f RMSE' % (testScore)) 
        
        trainSize = len(x_train)
        SampleSize = len(x_sample)
        TestSize = len(x_test)
        PredictSize = SampleSize-TestSize
        x_actual = x_sample[:TestSize]
        x_predict = testPredict
        print(SampleSize,TestSize,PredictSize)

        
        for j in range(PredictSize-1):
            if len(x_actual) == PredictSize:
                model, loss_history=neuralNetwork.myBiRNN(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,gru_units=gru_units,\
                                                      input_shape=input_shape,\
                                                      batch_size=batch_size,epochs=epochs)            
            else:
                model, loss_history=neuralNetwork.myBiRNN(x_train=x_train,y_train=y_train,x_test=x_actual1,y_test=x_predict,gru_units=gru_units,\
                                                      input_shape=input_shape,\
                                                      batch_size=batch_size,epochs=epochs)
            x_new=[x_predict[-1]]
            x_actual = x_sample[:TestSize+j]
            x_actual=np.append(x_actual,x_new,axis=0)
            x_actual1=scaler.transform(x_actual)
            #print(len(x_actual1))
            x_actual1=np.reshape(x_actual1,(len(x_actual1),1,1))

            x_predict=model.predict(x_actual1)
            j=j+1

        print(x_actual,x_predict)
        print(len(x_actual))
        print(len(x_predict))

        
        return x_actual,x_predict



    
