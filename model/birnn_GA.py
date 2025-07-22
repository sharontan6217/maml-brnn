#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 11:05:30 2019

@author: sharontan
"""

#!/usr/bin/env python
# coding: utf-8

"""
Train a LIBOR Preitctor AI using Genetic algorithms based BRNN.
"""


import sklearn.metrics
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional, BatchNormalization, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plotnine import *
import plotnine



# init data

PopulationSize = 215
PredictSize = 15
crossover_prob = 0.6
mutation_power = 0.015
targetRate=0.015
evaluation_rate=0.012
currentTime=datetime.datetime.now()

class neuralNetwork():
    
    def _init_(self,x_actual=None,x_predict=None,Diff=None,g=None,reward=None):    
        
        #reset
        self.x_test=x_test
        self.y_test=y_test
        self.x_predict=x_predict
        self.x_train=x_train
        self.y_train=y_train
        self.change=change
        self.Diff=x_predict-x_actual
        self.g=g
        self.fitness=fitness
        self.reward=reward
        self.x_new=x_new
        self.y_new=y_new
        self.x_sample=x_sample
        self.gru_units=gru_units
        self.dense_units=dense_units
        self.input_shape=input_shape
        self.batch_size=batch_size
        self.epochs=epochs
        self.drop_out=drop_out
        self.patience=patience
        
    
    def myBiRNN(self,
                x_train,
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
        
        #early_stopping=EarlyStopping(monitor="var_loss", patience=patience)
        model.compile(optimizer='adamax',loss='mean_squared_error',metrics=['mae'])
        
        history_callback=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,\
                                   verbose=2, validation_data=[x_test,y_test],shuffle = True)
        
        return model, history_callback
        
    
    def dataLoad():
        global scaler,n, startDate, predictDate, std, mean
        
        df=read_csv(data_dir,na_values='null',na_filter=True)
        df['USD1MTD156N']=pd.to_numeric(df['USD1MTD156N'],errors='coerce')
        df['DATE']=df['DATE'].astype(str)


        n=random.randint(PopulationSize,260)
        #n=300
        df=df.iloc[:,0:2]
        df=df.loc[pd.notnull(df['USD1MTD156N'])]
        df_train=df.iloc[-1001-n:-n,1:2]
        #print(df_train)

        
        #df_test=df.iloc[-215-n-1:-PredictSize-n,0:2]
        df_test=df.iloc[-n-1:-n+PopulationSize-PredictSize,0:2]
        df_test.reset_index()
        #print(df_test)
        startDate=df_test.iloc[0]['DATE']
        predictDate=df_test.iloc[200]['DATE']
        #print(startDate,predictDate)
        df_test=df_test.iloc[:,1:2]
        df_actual=df.iloc[-n-1:-n+PopulationSize,1:2]

        data=df_train.values
        #print(data)
        data_test=df_test.values
        data_actual=df_actual.values
        mean = np.mean(data_actual,axis=0)
        std = np.std(data_actual,axis=0)
        count=df_train.count()
        x_train=data[0:1000]
        y_train=data[1:1001]
        x_test=data_test[0:200]
        y_test=data_test[1:201]
        x_sample=data_actual[0:215]
        y_sample=data_actual[1:216]
        
        #Normalization
        scaler=MinMaxScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        
        x_train=np.reshape(x_train,(len(x_train),1,1))
        x_test=np.reshape(x_test,(len(x_test),1,1))
        
        
        print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test),np.count_nonzero(x_sample))
        return x_train,y_train,x_test,y_test,x_sample,y_sample
    
        
    def train(self):
        
        batch_size=128
        epochs=3000
        #drop_out=0.1
        #patience=5
        gru_units=90
        #dense_units=10
        input_shape=(None,1)
        
        #data load
        print("#data load:")
        x_train,y_train,x_test,y_test,x_sample,y_sample = neuralNetwork.dataLoad()
        print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test),np.count_nonzero(x_sample),np.count_nonzero(y_sample))
        
        
        #train
        model, loss_history=neuralNetwork.myBiRNN(self=self,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,gru_units=gru_units,\
                                                  input_shape=input_shape,\
                                                  batch_size=batch_size,epochs=epochs)
        
        trainPredict=model.predict(x_train)
        trainScore=math.sqrt(mean_squared_error(y_train,trainPredict)) 
        print('Train Score: %.5f RMSE' % (trainScore))
        
        testPredict=model.predict(x_test)
        testScore=math.sqrt(mean_squared_error(y_test,testPredict)) 
        print('Train Score: %.5f RMSE' % (testScore)) 
        
        realSize=PopulationSize-PredictSize
        x_actual = x_sample[0:200]
        x_predict = testPredict
        
        for j in range(PredictSize):
            
            if len(x_actual) == 200:
                model, loss_history=neuralNetwork.myBiRNN(self=self,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,gru_units=gru_units,\
                                                      input_shape=input_shape,\
                                                      batch_size=batch_size,epochs=epochs)            
            else:
                model, loss_history=neuralNetwork.myBiRNN(self=self,x_train=x_train,y_train=y_train,x_test=x_actual1,y_test=x_predict,gru_units=gru_units,\
                                                      input_shape=input_shape,\
                                                      batch_size=batch_size,epochs=epochs)
            x_new=[x_predict[realSize+j-1]]
            x_actual=np.append(x_actual,x_new,axis=0)
            x_actual1=scaler.transform(x_actual)
            #print(len(x_actual1))
            x_actual1=np.reshape(x_actual1,(len(x_actual1),1,1))

            x_predict=model.predict(x_actual1)
            j=j+1

        print(x_actual,x_predict)

       #generator
        g=[]
        reward=[]
        change=x_predict-x_actual
        for i in range(PopulationSize):
            change[i]=x_predict[i]-x_actual[i]
            if change[i]<0:
                g.append((-1)*log10((-1)*change[i]))
            if change[i]==0:
                g.append(0)
            if change[i]>0:
                g.append(log10(change[i]))
            reward.append(0)
            i=i+1
        #print(np.count_nonzero(x_actual))
        #print (x_actual,x_predict,g,reward,x_sample) 
        print("Test Dataset starts from ",n, "Test Dataset ends to ",n+PopulationSize, "Test Dataset StartDate",startDate,"Predict StartDate ",predictDate)
 
        
        return (x_actual,x_predict,g,reward,x_sample,y_sample,predictDate)  



    
