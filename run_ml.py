import pandas as pd
import numpy as np
import os
import torch
from torch import nn, optim
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import random
import model
from model import nn_sklearn, regression,tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import argparse
import matplotlib.pyplot as plt
import gc
from framework import searchCV, selftraining_ml
scaler_1 = MinMaxScaler()
scaler = StandardScaler()


def dataLoad():
    df_orig = pd.read_csv(data_path,header=0,na_filter=True)    
    df = df_orig[df_orig['ELEMENT']=='TMIN']
    df = df.replace(-9999,np.nan)
    df = df.dropna()
    df = df.drop(['COOP ID','YEAR','MONTH','ELEMENT'],axis=1)
    df = df.reset_index()
    data = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            col = df.columns[j]
            if 'value' in col.lower():
                #print(col)
                value = df.loc[i,col]
                data.append(value)
    orig = data[-poolSize-1:-1]
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return df,orig
def dataSplit(orig,trainSize,testSize,predictSize):
    global start
    '''
    x_train = np.array(x[-start:-start+trainSize])
    y_train = np.array(y[-start:-start+trainSize])
    x_test = np.array(x[-start+trainSize:-start+trainSize+testSize])
    y_test = np.array(y[-start+trainSize:-start+trainSize+testSize])
    y_actual = np.array(y[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize])
    '''
    mask = datamask(orig)
    df_imts = pd.DataFrame()
    df_imts['mask']=mask
    df_imts =df_imts[df_imts['mask']!=-1].reset_index()
    print(df_imts)
    start = random.randint(totalSize+1,len(df_imts)-totalSize)
    x = df_imts['mask'][-start-1:-1]
    y = df_imts['mask'][-start:]
    #print(len(total))
    #print(x,y)

    #print(len(x_total))
    #print(len(y_total))
    x_train = np.array(x[:trainSize]).reshape(-1,1)
    y_train = np.array(y[:trainSize]).reshape(-1,1)
    x_test = np.array(x[trainSize:trainSize+testSize]).reshape(-1,1)
    y_test = np.array(y[trainSize:trainSize+testSize]).reshape(-1,1)
    y_actual = np.array(y[trainSize+testSize:trainSize+testSize+predictSize]).reshape(-1,1)

    x= np.array(x).reshape(-1,1)
    y= np.array(y).reshape(-1,1)
    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    #print(x_train,y_train,x_test,y_test,y_actual)
    
    #print(y_train)
    #x_actual = np.array(x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)

    return x,y,x_train,y_train,x_test,y_test,y_actual
def datamask(data):
    y_train_mask = data
    y_mask = np.random.rand(len(data)) < 0.35
    
    #print(len(y_mask))
    count = 0
    for i in range(len(y_mask)):
        if y_mask[i] == True:
            y_train_mask[i] = -1
            count +=1
    return y_train_mask
def base_model(opt):
    model_name = opt.learning_name
    min_samples=testSize
    if model_name.lower() == 'RANSACRegressor'.lower():
        model_base, model_base_name = regression.myRegression.RANSACRegressor(min_samples)
    elif model_name.lower() == 'DecisionTree'.lower():
        model_base, model_base_name=tree.myTree.DecisionTree()
    elif model_name.lower() == 'decisionRegressor'.lower():
        model_base, model_base_name=tree.myTree.decisionRegressor()
    elif model_name.lower() == 'RandomForest'.lower():
        model_base, model_base_name = tree.myTree.RandomForest() 
    elif model_name.lower() == 'BernoulliRBM'.lower():  
        model_base, model_base_name=nn_sklearn.myNN.BernoulliRBM(min_samples)
    elif  model_name.lower() == 'MLP'.lower():
        model_base, model_base_name=nn_sklearn.myNN.MLP()
    elif  model_name.lower() == 'LogisticRegression'.lower():
        model_base, model_base_name = regression.myRegression.LogisticRegression()   
    elif  model_name.lower() == 'LinearRegression'.lower(): 
        model_base, model_base_name = regression.myRegression.LinearRegression()  
    elif  model_name.lower() == 'RidgeClassifier'.lower(): 
        model_base, model_base_name = regression.myRegression.RidgeClassifier()      
    elif  model_name.lower() == 'PassiveAggressiveRegressor'.lower(): 
        model_base, model_base_name = regression.myRegression.PassiveAggressiveRegressor()   
    elif  model_name.lower() == 'SGDRegressor'.lower(): 
        model_base, model_base_name = regression.myRegression.SGDRegressor()  
    elif  model_name.lower() == 'SGDClassifier'.lower(): 
        model_base, model_base_name = regression.myRegression.SGDClassifier()  
    elif  model_name.lower() == 'IsotonicRegression'.lower(): 
        model_base, model_base_name = regression.myRegression.IsotonicRegression()  
    elif  model_name.lower() == 'TheilSenRegressor'.lower(): 
        model_base, model_base_name = regression.myRegression.TheilSenRegressor() 

    return model_base,model_base_name
def selfTrainingModel(model_base,opt):
    training_model = opt.selftraining_name
    if training_model.lower() == 'selfTrainingClassifier'.lower():
        model_selfTraining,model_selfTraining_name = selftraining_ml.selfTrainingClassifier.selfTrainingClassifier(model_base)
    else:
        model_selfTraining,model_selfTraining_name = selftraining_ml.selfTrainingClassifier.LabelSpreading()

    return model_selfTraining,model_selfTraining_name
def scale(data):
    scaler_models=['SGDRegressor','SGDClassifier','BernoulliRBM']
    if model_base_name in scaler_models:
        data_scaled = scaler.fit_transform(data)
    elif model_base_name == 'MLP':
        data_scaled = scaler_1.fit_transform(data)
    else:
        data_scaled = data
    return data_scaled

def predict(x_train,y_train,x_test,y_test,y_actual,opt):
    print(opt.with_selftraining)
    print(model_base_name)


    if opt.with_selftraining == True:
        y_train_mask = datamask(y_train)
        y_predict, y_actual = train_daily(x_train,y_train_mask,x_test,y_test,y_actual,model_selfTraining,opt)   
    else:
        y_predict, y_actual = train_daily(x_train,y_train,x_test,y_test,y_actual,model_base,opt)
        #x_predict, y_test = train(x_train,y_train,x_test,y_test,y_actual,model_predict)
    return y_predict,y_actual
def meta_frame(model_base,opt):
    training_model = opt.metalearning_name
    if training_model.lower() == 'gridSearchCV'.lower():
        model_searchCV,model_searchCV_name = searchCV.searchCV.gridSearchCV(model_base)
    else:
        model_searchCV,model_searchCV_name = searchCV.searchCV.halvingGridSearchCV(model_base)
    return model_searchCV,model_searchCV_name

def train_daily(x_train,y_train,x_test,y_test,y_actual,model_predict,opt):
    #print(x_train)
    x_train = scale(x_train)
    x_test = scale(x_test)
    if opt.with_metalearning == True:
        model_searchCV,model_searchCV_name = meta_frame(model_base,opt)
        model_predict =model_searchCV
    #print('y_train is: ',y_train)
    x_total = np.concat((x_train,x_test),axis=0)
    y_total = np.concat((y_train,y_test),axis=0) 
    #print(x_total)
    #print(y_total)
    y_predict=[]
    for j in range(predictSize):
        if j==0:
            print('-----------j=0-----------------')
            model_predict.fit(x_train, y_train)
            score = model_predict.score(x_train,y_train)
            print('score is: ',score)
            x_predict = model_predict.predict(x_test)
        else:
            print('-----------j={}-----------------'.format(j))


            print(len(x_actual),len(y_actual1))
            model_predict.fit(x_actual, y_actual1)
            score = model_predict.score(x_actual, y_actual1)
            print('score is: ',score)
            x_predict = model_predict.predict(x_actual)
        x_new=x_predict[-1].reshape(-1,1)
        x_actual1 = x[j:trainSize+testSize+j-1]
        y_actual1 = y[j:trainSize+testSize+j]
        x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
        x_actual = scale(x_actual)
        y_predict.append(x_new)
        j=j+1
    print(y_predict)
    print(len(y_predict))
    y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    print('start is: ',start)
    return y_predict, y_actual
def train(x_train,y_train,x_test,y_test,y_actual,model_predict):

    model_predict.fit(x_train, y_train)
    score = model_predict.score(x_train,y_train)
    print('score is: ',score)
    x_predict = model_predict.predict(x_test)
    print('original data is: ')
    print(y_test)
    print('prediction is: ')
    print(x_predict)
    return x_predict, y_test

def evaluation(actual,predict):
    f1score = f1_score(actual.astype('int32'),predict.astype('int32'),average='micro')
    accuracy = accuracy_score(actual.astype('int32'),predict.astype('int32'))
    mse = mean_squared_error(actual,predict)
    mae = mean_absolute_error(actual,predict)
    with open (log_dir+'result_'+model_base_name+'.log','a') as f:
        f.write('F Measure={}\n'.format(f1score))
        f.write('Accuracy Score={}\n'.format(accuracy))
        f.write('mse={}\n'.format(mse))
        f.write('mae={}\n'.format(mae))
        f.close()
    return f1score,accuracy,mse,mae
def visualize(actual,predict):
    fig=plt.figure()
    plt.plot(actual,color='blue',label='Actual')
    plt.plot(predict,color='red',label = 'Prediction')
    plt.xlabel('Days')
    plt.ylabel('TMIN')
    plt.title('Plot Graph of Actual and Predicted TMIN')
    plt.legend(loc='best')
    fig_name='test_scenario_'+model_base_name+'_'+timeSequence+'_'+str(start)+'.png'

    plt.savefig(graph_dir+fig_name)
    plt.close()      

    return fig
def output(actual,predict):
    df_result=pd.DataFrame()
    df_result['actual']=actual
    df_result['predict']=predict
    output_name = 'output_'+model_base_name+'_'+timeSequence+'_'+str(start)+'.csv'
    df_result.to_csv(output_dir+output_name)
    return df_result
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_name',type=str,default='LinearRegression',help = 'learning methods is one of the list ["DecisionTreeRegressor", "LinearRegression","MLP","BernoulliRBM","PassiveAggressiveRegressor","RandomForest","RANSARegressor","RidgeClassifier","SGDRegressor","TheilSenRegressor","LogisticRegression","SGDClassifier","DecisionTree","decisionRegressor","RandomForest"]')
    parser.add_argument('--with_selftraining',type=bool,default=False, help = 'Defult to be False, True if adding self-training method.' )
    parser.add_argument('--selftraining_name',type=str,default='None', help = 'learning method is one of the list ["None", "selfTrainingClassifier","LabelSpreading"]')
    parser.add_argument('--with_metalearning',type=bool,default=False, help = 'Defult to be False, True if adding meta-learning method.' )
    parser.add_argument('--metalearning_name',type=str,default='None', help = 'learning method is one of the list ["None", "gridSearchCV","halvingGridSearchCV"]')
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    graph_dir = './graph/100-20/'
    log_dir = './log/100-20/'
    output_dir = './output/100-20/'

    poolSize = 1080
    trainSize = 100
    testSize = 20
    predictSize = 5
    totalSize = trainSize+testSize+predictSize
    opt = get_parser()
    df,orig = dataLoad()
    model_base,model_base_name = base_model(opt)
    if opt.with_metalearning==True:
        model_metalearning_name = opt.metalearning_name
        if opt.with_selftraining == True:
            model_selfTraining,model_selfTraining_name = selfTrainingModel(model_base,opt)
            log_dir = log_dir+model_metalearning_name+'/'+model_selfTraining_name+'/'+model_base_name+'/'
            output_dir = output_dir+model_metalearning_name+'/'+model_selfTraining_name+'/'+model_base_name+'/'
            graph_dir = graph_dir+model_metalearning_name+'/'+model_selfTraining_name+'/'+model_base_name+'/'
        else:
            log_dir = log_dir+model_metalearning_name+'/'+model_base_name+'/'
            output_dir = output_dir+model_metalearning_name+'/'+model_base_name+'/'
            graph_dir = graph_dir+model_metalearning_name+'/'+model_base_name+'/'
    else:
        if opt.with_selftraining == True:
            model_selfTraining,model_selfTraining_name = selfTrainingModel(model_base,opt)
            log_dir = log_dir+model_selfTraining_name+'/'+model_base_name+'/'
            output_dir = output_dir+model_selfTraining_name+'/'+model_base_name+'/'
            graph_dir = graph_dir+model_selfTraining_name+'/'+model_base_name+'/'
        else:
            log_dir = log_dir+model_base_name+'/'
            output_dir = output_dir+model_base_name+'/'
            graph_dir = graph_dir+model_base_name+'/'
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    try:
        for i in range(50):
            timeSequence = str(datetime.datetime.now())[20:26]
            #start = random.randint(totalSize,poolSize-1)
            #sampleSize = 15
            x,y,x_train,y_train,x_test,y_test,y_actual = dataSplit(orig,trainSize,testSize,predictSize)
            y_predict,y_actual = predict(x_train,y_train,x_test,y_test,y_actual,opt)
            #y_actual = y_actual[-predictSize-1:-1]
            #y_predict = y_predict[-predictSize-1:-1]

            f1score,accuracy,mse,mae = evaluation( y_actual,y_predict)
            print(f1score,accuracy,mse,mae )
            x_actual_ = np.squeeze( y_actual)
            x_predict_ = np.squeeze(y_predict)
            fig = visualize(x_actual_,x_predict_)  

            df_result = output(x_actual_,x_predict_)
            i+=1
    except Exception as e:
        print(e)
        with open('./error.log','a') as f:
            f.write(str(e))
            f.close

