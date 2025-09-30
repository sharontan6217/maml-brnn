import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import model
from model import brnn 
from model.brnn import neuralNetwork
import keras
import math
import argparse
import datetime
import random
import time
import matplotlib.pyplot as plt
from framework import reptile,maml
import gc

scaler = StandardScaler()
def eeg_dataLoad():
    global cols_orig
    df_orig = pd.read_csv(data_path,header=0,na_filter=True)  
    print(df_orig.columns) 
    df = df_orig[['P4','Cz','F8','T7']]
    cols_orig = df.columns
    df = df.replace(np.nan,-1)

    '''
    data = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            col = df.columns[j]
            if 'f8' in col.lower():
                value = df.loc[i,col]
                data.append(value)
    '''
    orig = df[-poolSize:]
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig

def dataSplit(orig):
    '''
    x_train = np.array(x[-start:-start+trainSize])
    y_train = np.array(y[-start:-start+trainSize])
    x_test = np.array(x[-start+trainSize:-start+trainSize+testSize])
    y_test = np.array(y[-start+trainSize:-start+trainSize+testSize])
    y_actual = np.array(y[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize])
    '''
    print(len(orig))
    mask = datamask(orig)
    #df_imts = pd.DataFrame()
    df_imts=mask
    '''
    for col in df_imts.columns:
        df_imts =df_imts[df_imts[col]!=-1 ]
    '''
    #print(df_imts)
    #df_imts = df_imts.drop(('index'),axis=1)
    x = df_imts[-start-1:-1]
    y = df_imts[-start:]
    x = x.reset_index()
    y = y.reset_index()
    x_imputate = imputate(x,imputated_value=-1)
    y_imputate = imputate(y,imputated_value=-1)
    #print(len(total))
    print('------------imputated x is------------------')
    print(x_imputate)
    x = x.drop(('index'),axis=1)
    y = y.drop(('index'),axis=1)
    if 'index' in orig.columns:
        orig = orig.drop(('index'),axis=1)
    y_orig = orig[-start:]
    #print(len(x_total))
    #print(len(y_total))
    x_train = np.array(x_imputate[:trainSize])
    y_train = np.array(y_imputate[:trainSize])
    x_test = np.array(x_imputate[trainSize:trainSize+testSize])
    y_test = np.array(y_imputate[trainSize:trainSize+testSize])
    y_actual = np.array(y_orig[trainSize+testSize:trainSize+testSize+predictSize])
    x= np.array(x)
    y= np.array(y)
    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    #print(x_train,y_train,x_test,y_test,y_actual)
    
    #print(y_train)
    #x_actual = np.array(x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)

    return x,y,x_imputate,y_imputate,x_train,y_train,x_test,y_test,y_actual

def datamask(data):
    y_train_mask = data
    print(y_train_mask)
    
    data = data.reset_index()
    #print(len(y_mask))
    count = 0
    for col in y_train_mask.columns:
        y_mask = np.random.rand(len(data)) < 0.2
        for i in range(len(y_mask)):
            if y_mask[i] == True:
                y_train_mask.loc[i,col] = -1
                count +=1
            i+=1
    data = data.drop(('index'),axis=1)
    print(data)
    return data
def logarithm(x_orig,y_orig):
    diff =y_orig-x_orig
    orig_log=[]
    for i in range(len(x_orig)):
        diff[i]=y_test[i]-x_test[i]
        if diff[i]<0:
            orig_log.append((-1)*log2((-1)*diff[i]))
        if diff[i]==0:
            orig_log.append(0)
        if diff[i]>0:
            orig_log.append(log2(diff[i]))
        i+=1
    return orig_log
def reverse_logarithm(orig_log):
    diff=np.zeros(shape=(orig_log.shape[0],1))
    for i in range(orig_log.shape[0]):
        if orig_log[i, :]>0:
            diff[i]=(-1)*np.exp2((-1)*orig_log[i, :])
        if orig_log[i, :]==0:
            diff[i]=0
        if orig_log[i, :]<0:
            diff[i]=np.exp2(orig_log[i, :])
    return diff
def meta_convertion(data_meta):
    df_meta = pd.DataFrame(data=data_meta)
    data_converted = []
    for col in df_meta.columns:
        data_converted.append(df_meta[col])
    data_converted = np.asarray(data_converted).reshape(-1,1,1)
    return data_converted
def imputate(df,imputated_value):
    if 'index' in df.columns:
        df = df.drop(('index'),axis=1)
    df = df.reset_index()
    for col in df.columns:
        for i in range(len(df)):
            try:
                print(i,col)
                if df.loc[i,col]==imputated_value:
                    if df.loc[i-1,col]!=imputated_value and df.loc[i+1,col]!=imputated_value:
                        df.loc[i,col] = np.mean([0.8*df.loc[i-1,col],1.2*df.loc[i+1,col]])
                    else:
                        estimate_matrix = [item for item in df.loc[i-8:i-1,col] if item!=imputated_value]
                        df.loc[i,col]  =np.mean(estimate_matrix)
                else:
                    continue
            except Exception as e:
                    #print(e)
                    df.loc[i,col]  =np.mean(df.loc[i-8:i-1,col])
    df = df.drop(('index'),axis=1)
    return df


def selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_test,y_test):
    training_model = opt.selftraining_name
    if training_model.lower() == 'selfTrainingClassifier'.lower():
        y_predict,model_selfTraining,model_selfTraining_name  = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train_mask,x_test,y_test)
    else:
       y_predict,model_selfTraining,model_selfTraining_name = selfTrainingClassifier.LabelSpreading(model_predict,x_train,y_train,x_test,y_test)

    return y_predict

def meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt):
    gc.collect()
    global len_cols
    len_cols = len(cols_orig)
    meta_optimizer = keras.optimizers.Adam(learning_rate=0.005)
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    meta_name = opt.metalearning_name
    x_train=scaler.fit_transform(x_train)
    x_train_meta = meta_convertion(x_train)
    x_test=scaler.transform(x_test)
    x_test_meta = meta_convertion(x_test)
    y_train=scaler.fit_transform(y_train).astype(np.float32)
    y_train_meta = meta_convertion(y_train)
    y_test=scaler.transform(y_test).astype(np.float32)
    y_test_meta = meta_convertion(y_test)
    y_actual=scaler.transform(y_actual).astype(np.float32)
    #num_tasks = int(np.round((trainSize/testSize)*len_cols))
    num_tasks = int(np.round(trainSize/testSize))
    #num_tasks = 2
    #x_train=np.reshape(x_train,(len(x_train),1,1))
    #x_test=np.reshape(x_test,(len(x_test),1,1))
    # Dummy tasks for demonstration
    for j in range(predictSize):
        if opt.with_selftraining == True:
            if j==0:
                # Train the model using MAML
                tasks = [
                {'X': x_train_meta[:trainSize], 'Y':  y_train_meta[:trainSize]},
                {'X': x_train_meta[trainSize:trainSize*2], 'Y':  y_train_meta[trainSize:trainSize*2]},
                {'X': x_train_meta[trainSize*2:trainSize*3], 'Y':  y_train_meta[trainSize*2:trainSize*3]},
                {'X': x_train_meta[trainSize*3:], 'Y':  y_train_meta[trainSize*3:]},
                {'X': x_test_meta[:testSize], 'Y':  y_test_meta[:testSize]},
                {'X': x_test_meta[testSize:testSize*2], 'Y':  y_test_meta[testSize:testSize*2]},
                {'X': x_test_meta[testSize*2:testSize*3], 'Y':  y_test_meta[testSize*2:testSize*3]},
                {'X': x_test_meta[testSize*3:], 'Y':  y_test_meta[testSize*3:]}]
                maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.01, meta_steps=100, inner_steps=5)
                #y_predict = model_predict.predict(x_test)
                y_predict=selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_test,y_test)
            else:
                tasks = [
                {'X': x_train_meta[:trainSize], 'Y':  y_train_meta[:trainSize]},
                {'X': x_train_meta[trainSize:trainSize*2], 'Y':  y_train_meta[trainSize:trainSize*2]},
                {'X': x_train_meta[trainSize*2:trainSize*3], 'Y':  y_train_meta[trainSize*2:trainSize*3]},
                {'X': x_train_meta[trainSize*3:], 'Y':  y_train_meta[trainSize*3:]},

                {'X': x_actual_meta[:testSize], 'Y':  y_predict[:testSize]},
                {'X': x_actual_meta[testSize:testSize*2], 'Y':  y_predict[testSize:testSize*2]},
                {'X': x_actual_meta[testSize*2:testSize*3], 'Y':  y_predict[testSize*2:testSize*3]},
                {'X': x_actual_meta[testSize*3:], 'Y':  y_predict[testSize*3:]}]

                maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.005, meta_steps=100, inner_steps=5)
                y_predict=selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_actual_meta,y_predict)
            #y_predict = model_predict.predict(x_actual)
            df_predict = pd.DataFrame()
            df_predict['P4']=np.array(y_predict).reshape(-1,)[:testSize]
            df_predict['Cz']=np.array(y_predict).reshape(-1,)[testSize:testSize*2]
            df_predict['F8']=np.array(y_predict).reshape(-1,)[testSize*2:testSize*3]
            df_predict['T7']=np.array(y_predict).reshape(-1,)[testSize*3:]
            x_new=df_predict.values[-1].reshape(1,-1)
            x_actual1 = x_imputate[trainSize+j:trainSize+testSize+j-1]
            print(x_new)
            print(x_actual1)
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual_meta = meta_convertion(x_actual)
            
            #x_actual=np.reshape(x_actual,(len(x_actual),len_cols,1))
            #x_new=x_predict[-1].reshape(-1,1)
            #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
            #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
            #y_predict.append(x_new)
        else:
            if j==0:
                # Train the model using MAML
                train_tasks = []
                test_tasks=[]
                for n in range(num_tasks-1):
                    if n==0:
                        train_tasks_ = {'X': x_train_meta[:testSize], 'Y':  y_train_meta[:testSize]}
                        test_tasks_ =  {'X': x_test_meta[:testSize], 'Y':  y_test_meta[:testSize]}  
                    elif n==num_tasks-1:
                        train_tasks_ = {'X': x_train_meta[testSize*n:], 'Y':  y_train_meta[testSize*n:]} 
                        test_tasks_ = {'X': x_test_meta[testSize*n:], 'Y':  y_test_meta[testSize*n:]} 
                    else:
                        train_tasks_ = {'X': x_train_meta[testSize*n:testSize*(n+1)], 'Y':  y_train_meta[testSize*n:testSize*(n+1)]} 
                        test_tasks_ =  {'X': x_test_meta[testSize*n:testSize*(n+1)], 'Y':  y_test_meta[testSize*n:testSize*(n+1)]} 
                    
                    train_tasks.append(train_tasks_)      
                    test_tasks.append(test_tasks_)
                    n+=1
                tasks = np.concatenate((train_tasks,test_tasks),axis=0)         
                print(tasks)

                maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.01, meta_steps=200, inner_steps=5)
                y_predict = model_predict.predict(x_test_meta)
                print(j)
                print(y_predict)
                print(y_predict.shape)

            else:
                
                train_tasks = []
                test_tasks=[]
                for n in range(num_tasks-1):
                    if n==0:
                        train_tasks_ = {'X': x_train_meta[:testSize], 'Y':  y_train_meta[:testSize]}
                        test_tasks_ =  {'X': x_actual_meta[:testSize], 'Y':  y_predict[:testSize]}  
                    elif n==num_tasks-1:
                        train_tasks_ = {'X': x_train_meta[testSize*n:], 'Y':  y_train_meta[testSize*n:]} 
                        test_tasks_ = {'X': x_actual_meta[testSize*n:], 'Y':  y_predict[testSize*n:]} 
                    else:
                        train_tasks_ = {'X': x_train_meta[testSize*n:testSize*(n+1)], 'Y':  y_train_meta[testSize*n:testSize*(n+1)]} 
                        test_tasks_ =  {'X': x_actual_meta[testSize*n:testSize*(n+1)], 'Y':  y_predict[testSize*n:testSize*(n+1)]} 
                    
                    train_tasks.append(train_tasks_)      
                    test_tasks.append(test_tasks_)
                    n+=1
                tasks = np.concatenate((train_tasks,test_tasks),axis=0)       
                print(tasks)
                '''
                tasks = [
                {'X': x_train_meta[:trainSize], 'Y':  y_train_meta[:trainSize]},
                {'X': x_train_meta[trainSize:trainSize*2], 'Y':  y_train_meta[trainSize:trainSize*2]},
                {'X': x_train_meta[trainSize*2:trainSize*3], 'Y':  y_train_meta[trainSize*2:trainSize*3]},
                {'X': x_train_meta[trainSize*3:trainSize*4], 'Y':  y_train_meta[trainSize*3:trainSize*4]},
                {'X': x_train_meta[trainSize*4:], 'Y':  y_train_meta[trainSize*4:]},
                {'X': x_actual_meta[:testSize], 'Y':  y_predict[:testSize]},
                {'X': x_actual_meta[testSize:testSize*2], 'Y':  y_predict[testSize:testSize*2]},
                {'X': x_actual_meta[testSize*2:testSize*3], 'Y':  y_predict[testSize*2:testSize*3]},
                {'X': x_actual_meta[testSize*3:testSize*4], 'Y':  y_predict[testSize*3:testSize*4]},
                {'X': x_actual_meta[testSize*4:], 'Y':  y_predict[testSize*4:]}]
                '''
                maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.005, meta_steps=200, inner_steps=5)
                
                y_predict = model_predict.predict(x_actual_meta)
                print(j)
                print(y_predict)
                print(y_predict.shape)
            df_predict = pd.DataFrame()
            df_predict['P4']=np.array(y_predict).reshape(-1,)[:testSize]
            df_predict['Cz']=np.array(y_predict).reshape(-1,)[testSize:testSize*2]
            df_predict['F8']=np.array(y_predict).reshape(-1,)[testSize*2:testSize*3]
            df_predict['T7']=np.array(y_predict).reshape(-1,)[testSize*3:]
            x_new=df_predict.values[-1].reshape(1,-1)
            x_actual1 = x_imputate[trainSize+j:trainSize+testSize+j-1]
            print(x_new)
            print(x_actual1)
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual_meta = meta_convertion(x_actual)
            #x_actual=np.reshape(x_actual,(len(x_actual),1,1))
            #x_new=x_predict[-1].reshape(-1,1)
            #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
            #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
            #y_predict.append(x_new)
        j=j+1
    df_predict = pd.DataFrame()
    df_predict['P4']=np.array(y_predict).reshape(-1,)[:testSize]
    df_predict['Cz']=np.array(y_predict).reshape(-1,)[testSize:testSize*2]
    df_predict['F8']=np.array(y_predict).reshape(-1,)[testSize*2:testSize*3]
    df_predict['T7']=np.array(y_predict).reshape(-1,)[testSize*3:]
    print(len(df_predict))
    print(df_predict)
    y_predict = df_predict.values
    y_predict = y_predict[-predictSize:]
    print(len(y_predict))
    #y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    return y_predict, y_actual,model_predict
def train(model_predict,x_train,y_train,x_test,y_test,y_actual):
    gc.collect()


    #data load
    print("#data load:")
    print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test))
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    #train

    loss_history=model_predict.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_test,y_test])        
    y_predict=[]
    for j in range(predictSize):
        if opt.with_selftraining == True:
            if j == 0:
                print(j)
                y_predict_ = selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_test,y_test)    
            else:
                loss_history=model_predict.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_actual,y_predict_]) 
            x_new=y_predict_[-1].reshape(-1,1)
            x_actual1 = x_imputate[trainSize+j:trainSize+testSize+j-1]
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual=np.reshape(x_actual,(len(x_actual),1,1))
            y_predict_=selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_actual,y_predict_)
            #y_predict_=model_predict.predict(x_actual)
            print(y_predict_[-1][0])
            y_predict.append(y_predict_[-1][0])
        else:
            if j == 0:
                y_predict_ = model_predict.predict(x_test)          
            else:
                loss_history=model_predict.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_actual,y_predict_]) 
            x_new=y_predict_[-1].reshape(-1,1)
            x_actual1 = x_imputate[trainSize+j:trainSize+testSize+j-1]
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual=np.reshape(x_actual,(len(x_actual),1,1))
            y_predict_=model_predict.predict(x_actual)
            print(y_predict_[-1][0])
            y_predict.append(y_predict_[-1][0])
        j=j+1
    y_predict = np.array(y_predict).reshape(-1,1)
    #y_predict = y_predict[-predictSize-1:-1]

        
    return  y_predict, y_actual,model_predict

def evaluation(actual,predict):
    f1score = f1_score(actual.astype('int32'),predict.astype('int32'),average='micro')
    accuracy = accuracy_score(actual.astype('int32'),predict.astype('int32'))
    mse = mean_squared_error(actual,predict)
    mae = mean_absolute_error(actual,predict)
    with open (log_dir+'result.log','a') as f:
        f.write('F Measure={}\n'.format(f1score))
        f.write('Accuracy Score={}\n'.format(accuracy))
        f.write('mse={}\n'.format(mse))
        f.write('mae={}\n'.format(mae))
        f.close()
    return f1score,accuracy,mse,mae
def visualize(actual,predict):
    predict_0=predict[:predictSize]
    predict_1=predict[predictSize:predictSize*2]
    predict_2=predict[predictSize*2:predictSize*3]
    predict_3=predict[predictSize*3:]

    actual_0=actual[:predictSize]
    actual_1=actual[predictSize:predictSize*2]
    actual_2=actual[predictSize*2:predictSize*3]
    actual_3=actual[predictSize*3:]


    fig=plt.figure(1)
    plt.plot(actual_0,color='blue',marker='o',label='Actual')
    plt.plot(predict_0,color='red',marker='o',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('P4')
    plt.title('Plot Graph of  EEG Collected Via P4 Channel')
    plt.legend(loc='best')
    fig_name='test_scenario_p4'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close()      
    fig=plt.figure(2)
    plt.plot(actual_1,color='blue',marker='o',label='Actual')
    plt.plot(predict_1,color='red',marker='o',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Cz')
    plt.title('Plot Graph of Actual and Predicted EEG Collected Via Cz Channel')
    plt.legend(loc='best')
    fig_name='test_scenario_cz'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close() 
    fig=plt.figure(3)
    plt.plot(actual_2,color='blue',marker='o',label='Actual')
    plt.plot(predict_2,color='red',marker='o',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('F8')
    plt.title('Plot Graph of Actual and Predicted EEG Collected Via F8 Channel')
    plt.legend(loc='best')
    fig_name='test_scenario_f8'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close() 
    fig=plt.figure(4)
    plt.plot(actual_3,color='blue',marker='o',label='Actual')
    plt.plot(predict_3,color='red',marker='o',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Wind_direction')
    plt.title('Plot Graph of Actual and Predicted EEG Collected Via T7 Channel')
    plt.legend(loc='best')
    fig_name='test_scenario_t7'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close()  
    return fig
def output(actual,predict):
    df_result=pd.DataFrame()
    df_result['actual']=actual
    df_result['predict']=predict
    output_name = 'output_'+timeSequence+'_'+str(start)+'.csv'
    df_result.to_csv(output_dir+output_name)
    return df_result
class selfTrainingClassifier():
    def selfTrainingClassifier(model_predict,x_train,y_train_mask,x_test,y_test):
        model_selfTraining_name = 'selfTrainingClassifier'
        df_estimate=pd.DataFrame()
        df_estimate['x_train']=list(x_train.reshape(-1,))
        df_estimate['y_train']=list(y_train_mask.reshape(-1,))
        print(df_estimate)
        y_estimate = []
        for i in range(len(df_estimate)):
            try:
                if y_train_mask[i]==-1:
                    y_estimate_ = np.mean([0.8*y_train_mask[i-1],1.2*y_train_mask[i+1]])
                    y_estimate.append([y_estimate_])
                else:
                    y_estimate.append(y_train_mask[i])
                    
            except Exception as e:
                    print(e)
                    y_estimate_ = y_train_mask[i-1]
                    y_estimate.append(y_estimate_)
        df_estimate['y_estimate']=y_estimate
        print(df_estimate)
        df_labeleled = df_estimate[df_estimate['y_train']!=-1]
        X_labelled = np.array(df_labeleled['x_train']).reshape(-1,1,1)
        y_labelled = np.array(df_labeleled['y_train']).reshape(-1,1)
        #X_labelled=np.reshape(X_labelled,(len(X_labelled),1,1))

        loss_history=model_predict.fit(X_labelled,y_labelled,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=[x_test,y_test])
        confidence_threshold = 1
        for iteration in range(10):  # Run 5 iterations
            pseudo_y = model_predict.predict(x_train)  # Generate pseudo-labels
            pseudo_y = np.array(pseudo_y ).reshape(-1,1)
            y_estimate = np.array(y_estimate).reshape(-1,1)
            pseudo_probabilities = (pseudo_y-y_estimate).min(axis=1)  # Get confidence scores

            confident_indices = np.where(pseudo_probabilities < confidence_threshold)[0]  # Identify confident samples
            for indice in confident_indices:
                df_estimate.loc[indice,'y_train']=pseudo_y[indice]
            df_labeleled_ = df_estimate[df_estimate['y_train']!=-1]
            X_labelled = np.array(df_labeleled_['x_train']).reshape(-1,1,1)
            y_labelled = np.array(df_labeleled_['y_train']).reshape(-1,1)
            # Retrain the model with the expanded labeled dataset
            model_predict.fit(X_labelled, y_labelled)
        # Predict labels on the test dataset
        y_predict = model_predict.predict(x_test)

        # Print accuracy
        #print("Final Model Accuracy on Test Data:", accuracy_score(y_test, y_predict))
        return y_predict,model_predict,model_selfTraining_name
    def LabelSpreading(model_predict,x_train,y_train,x_test,y_test):
        split_labelled = int(len(x_train)*0.8)
        X_labeled, y_labeled = x_train[:split_labelled], y_train[:split_labelled]
        
        X_unlabeled, y_unlabeled = x_train[split_labelled:], y_train[split_labelled:]
        print(len( X_unlabeled),len(y_unlabeled))
        model_predict.fit(X_labeled, y_labeled,batch_size=batch_size,epochs=epochs,verbose=2,  validation_data=[x_test,y_test])

        confidence_threshold = 2
        for iteration in range(5):  # Run 5 iterations
            pseudo_y = model_predict.predict(X_unlabeled)  # Generate pseudo-labels
            pseudo_probabilities = (pseudo_y-y_unlabeled).min(axis=1)  # Get confidence scores

            confident_indices = np.where(pseudo_probabilities < confidence_threshold)[0]  # Identify confident samples

            # Add confident pseudo-labeled samples to the labeled dataset
            X_labeled = np.vstack((X_labeled, X_unlabeled[confident_indices]))
            y_labeled = np.vstack((y_labeled, pseudo_y[confident_indices]))

            # Remove pseudo-labeled samples from the unlabeled set
            X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)

            # Retrain the model with the expanded labeled dataset
            model_predict.fit(X_labeled, y_labeled)

        # Predict labels on the test dataset
        y_predict = model_predict.predict(x_test)

        # Print accuracy
        #print("Final Model Accuracy on Test Data:", accuracy_score(y_test, y_predict))
        model_selfTraining_name = 'LabelSpreading'
        return y_predict,model_predict,model_selfTraining_name 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metalearning',type=bool,default=False, help = 'Defult to be False, True if adding meta-learning method.' )
    parser.add_argument('--metalearning_name',type=str,default='None', help = 'learning method is one of the list ["None", "reptile","MAML"], reptile for gradient decent algorithms and Model Agonistic Meta Learning (MAML) for ML and DL algorithms' )
    parser.add_argument('--with_selftraining',type=bool,default=False, help = 'Defult to be False, True if adding self-training method.' )
    parser.add_argument('--selftraining_name',type=str,default='None', help = 'learning method is one of the list ["None", "selfTrainingClassifier","LabelSpreading"]')
    opt = parser.parse_args()
    return opt
if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    data_path = 'C:/Users/smile/Documents/GitHub/random_few_shot/data/Physionet/eeg/s01_ex01_s01.csv'
    graph_dir = './graph/50-10/'
    log_dir = './log/50-10/'
    output_dir = './output/50-10/'

    batch_size=32
    epochs=3000
    drop_out=0.2
    patience=5
    gru_units=20
    dense_units=5
    input_shape=(None,1)
        
    
    poolSize = 1080
    trainSize = 50
    testSize = 10
    predictSize = 5
    totalSize = trainSize+testSize+predictSize
    opt = get_parser()
    orig = eeg_dataLoad()
  
    model_predict = neuralNetwork.myBiRNN(gru_units=gru_units,drop_out=drop_out,input_shape=input_shape)
    if opt.with_metalearning==True:
        model_metalearning_name = opt.metalearning_name
        if opt.with_selftraining == True:
            model_selfTraining_name = opt.selftraining_name
            log_dir = log_dir+model_metalearning_name+'/'+model_selfTraining_name+'/eeg/BRNN/'
            output_dir = output_dir+model_metalearning_name+'/'+model_selfTraining_name+'/eeg/BRNN/'
            graph_dir = graph_dir+model_metalearning_name+'/'+model_selfTraining_name+'/eeg/BRNN/'
        else:
            log_dir = log_dir+model_metalearning_name+'/eeg/BRNN/'
            output_dir = output_dir+model_metalearning_name+'/eeg/BRNN/'
            graph_dir = graph_dir+model_metalearning_name+'/eeg/BRNN/'
    else:
        if opt.with_selftraining == True:
            model_selfTraining_name = opt.selftraining_name
            log_dir = log_dir+model_selfTraining_name+'/eeg/BRNN/'
            output_dir = output_dir+model_selfTraining_name+'/eeg/BRNN/'
            graph_dir = graph_dir+model_selfTraining_name+'/eeg/BRNN/'
        else:
            log_dir = log_dir+'/eeg/BRNN/'
            output_dir = output_dir+'/eeg/BRNN/'
            graph_dir = graph_dir+'/eeg/BRNN/'
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    for i in range(1):
        time.sleep(10)
        gc.collect()
        start = random.randint(totalSize+1,len(orig)-1)
        timeSequence = str(datetime.datetime.now())[20:26]
        x,y,x_imputate,y_imputate,x_train,y_train,x_test,y_test,y_actual = dataSplit(orig)
        y_train_mask = datamask(y_imputate[:trainSize])
        y_predict, y_actual, model_predict = meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt)

        print('original data is: ')
        print(y_actual)
        print('prediction is: ')
        print(y_predict)
        y_predict_fl = meta_convertion(y_predict).reshape(-1,1)
        y_actual_fl = meta_convertion(y_actual).reshape(-1,1)
        print('original data is: ')
        print(y_actual)
        print('prediction is: ')
        print(y_predict)
        f1score,accuracy,mse,mae = evaluation(y_actual_fl,y_predict_fl)
        print(f1score,accuracy,mse,mae )
        x_actual_ = np.squeeze(y_actual_fl)
        x_predict_ = np.squeeze(y_predict_fl)
        print(x_actual_)
        fig = visualize(x_actual_,x_predict_)
        #fig = visualize(y_actual,y_predict)   

        df_result = output(x_actual_,x_predict_)



