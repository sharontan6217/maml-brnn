import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
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
    x_orig = data[-poolSize-1:-1]
    y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    print(len(data),len(x_orig),len(y_orig))
    return df,x_orig,y_orig
def dataSplit(x_orig,y_orig,start,trainSize,testSize,predictSize):
    '''
    x_train = np.array(x[-start:-start+trainSize])
    y_train = np.array(y[-start:-start+trainSize])
    x_test = np.array(x[-start+trainSize:-start+trainSize+testSize])
    y_test = np.array(y[-start+trainSize:-start+trainSize+testSize])
    y_actual = np.array(y[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize])
    '''
    idx = []
    total_=[]
    for n in range(trainSize+testSize+predictSize+1):
        number = random.randint(start,len(x_orig)-1)

        idx.append(number)
        total_.append(x_orig[number])
        n+=1
    df_imts = pd.DataFrame()
    df_imts['idx']=idx
    df_imts['total']=total_
    df_imts =df_imts.sort_values(by='idx',ascending=True).reset_index()
    print(df_imts)
    total = np.array(df_imts['total'].values).reshape(-1,1)
    #print(len(total))
    x = total[:-1]
    y = total[1:]
    #print(len(x_total))
    #print(len(y_total))
    x_train = x[:trainSize]
    y_train = y[:trainSize]
    x_test = x[trainSize:trainSize+testSize]
    y_test = y[trainSize:trainSize+testSize]
    y_actual = y[trainSize+testSize:trainSize+testSize+predictSize]

    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    print(x_train,y_train,x_test,y_test,y_actual)
    
    #print(y_train)
    #x_actual = np.array(x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)

    return x,y,x_train,y_train,x_test,y_test,y_actual
def meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt):
    meta_optimizer = keras.optimizers.AdamW(learning_rate=0.005)
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    meta_name = opt.metalearning_name
    #x_train=scaler.fit_transform(x_train)
    #x_test=scaler.transform(x_test)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    # Dummy tasks for demonstration
    if meta_name.lower()=='reptile'.lower():
        # Train the model using reptile
        tasks = [
        {'X': x_train, 'Y':  y_train},
        {'X': x_test, 'Y':  y_test}]
        training,testing,model_predict=reptile.reptile_train(model_predict, tasks, meta_optimizer, meta_steps=600)
        reptile.reptile_visualize(training,testing,timeSequence,start,graph_dir)
    elif meta_name.lower()=='MAML'.lower():
        # Train the model using reptile
        tasks = [
        {'X': x_train[:testSize], 'Y':  y_train[:testSize]},
        {'X': x_train[-testSize-1:-1], 'Y':  y_train[:testSize]},
        {'X': x_test, 'Y':  y_test}]
        maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.01, meta_steps=400, inner_steps=5)
    y_predict=[]
    for j in range(predictSize):
        if j==0:
            x_predict = model_predict.predict(x_test)
        else:
            x_predict = model_predict.predict(x_actual)
        x_new=x_predict[-1].reshape(-1,1)
        x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        x_actual = np.append(x_actual1,x_new,axis=0)
        #x_actual=scaler.transform(x_actual1)
        x_actual=np.reshape(x_actual,(len(x_actual),1,1))
        #x_new=x_predict[-1].reshape(-1,1)
        #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
        y_predict.append(x_new)
        j=j+1
    #y_predict = y_predict[-predictSize-1:-1]
    y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    print('start is: ',start)
    return y_predict, y_actual
def train(model_predict,x_train,y_train,x_test,y_test,y_actual):


    #data load
    print("#data load:")
    print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test))
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    #train

    trainPredict=model_predict.predict(x_train)
    trainScore=math.sqrt(mean_squared_error(y_train,trainPredict)) 
    print('Train Score: %.5f RMSE' % (trainScore))
        
    testPredict=model_predict.predict(x_test)

    testScore=math.sqrt(mean_squared_error(y_test,testPredict)) 
    print('Test Score: %.5f RMSE' % (testScore))
    y_predict = testPredict  
    for j in range(predictSize-1):
        if j == 0:
            loss_history=model_predict.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_test,y_test],shuffle = True)            
        else:
            loss_history=model_predict.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_actual,y_predict],shuffle = True) 
        x_new=y_predict[-1].reshape(-1,1)
        x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        x_actual = np.append(x_actual1,x_new,axis=0)
        x_actual=scaler.transform(x_actual1)
        x_actual=np.reshape(x_actual,(len(x_actual),1,1))
        y_predict=model_predict.predict(x_actual)
        j=j+1

    y_predict = y_predict[-predictSize-1:-1]

        
    return y_actual,y_predict

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
    fig=plt.figure()
    plt.plot(actual,color='blue',label='Actual')
    plt.plot(predict,color='red',label='Prediction')
    plt.xlabel('Days')
    plt.ylabel('TMIN')
    plt.title('Plot Graph of Actual and Predicted TMIN')
    plt.legend(loc='best')
    fig_name='test_scenario_'+timeSequence+'_'+str(start)+'_brnn.png'
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
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metalearning',type=bool,default=False, help = 'Defult to be False, True if adding meta-learning method.' )
    parser.add_argument('--metalearning_name',type=str,default='None', help = 'learning method is one of the list ["None", "reptile","MAML"], reptile for gradient decent algorithms and Model Agonistic Meta Learning (MAML) for ML and DL algorithms' )
    opt = parser.parse_args()
    return opt
if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    graph_dir = './graph/15-5/'
    log_dir = './log/15-5/'
    output_dir = './output/15-5/'

    batch_size=32
    epochs=3000
    drop_out=0.1
    patience=5
    gru_units=20
    dense_units=5
    input_shape=(None,1)
        
    
    poolSize = 3600
    trainSize = 10
    testSize = 5
    predictSize = 5
    totalSize = trainSize+testSize+predictSize
    opt = get_parser()
    df,x,y = dataLoad()
    model_predict = neuralNetwork.myBiRNN(gru_units=gru_units,drop_out=drop_out,input_shape=input_shape)
    if opt.with_metalearning==True:
        model_metalearning_name = opt.metalearning_name
        log_dir = log_dir+model_metalearning_name+'/BRNN/'
        output_dir = output_dir+model_metalearning_name+'/BRNN/'
        graph_dir = graph_dir+model_metalearning_name+'/BRNN/'
    else:
        log_dir = log_dir+'/BRNN/'
        output_dir = output_dir+'/BRNN/'
        graph_dir = graph_dir+'/BRNN/'
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    for i in range(5):
        time.sleep(10)
        start = random.randint(0,3500)
        timeSequence = str(datetime.datetime.now())[20:26]
        x_train,y_train,x_test,y_test,y_actual = dataSplit(x,y,start,trainSize,testSize,predictSize)
        if opt.with_metalearning==True:
            y_actual,y_predict = meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt)
        else:
            y_actual,y_predict = train(model_predict,x_train,y_train,x_test,y_test,y_actual)
            

        print('original data is: ')
        print(y_actual)
        print('prediction is: ')
        print(y_predict)
        f1score,accuracy,mse,mae = evaluation(y_actual,y_predict)
        print(f1score,accuracy,mse,mae )
        x_actual_ = np.squeeze(y_actual)
        x_predict_ = np.squeeze(y_predict)
        fig = visualize(x_actual_,x_predict_)  
        df_result = output(x_actual_,x_predict_)


