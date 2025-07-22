import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import model
from model import birnn_orig
import datetime
import random
import time
import matplotlib.pyplot as plt
import gc

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
    #start = len(df)-totalSize
    x = data[-start-totalSize-2:-start-2]
    y = data[-start-totalSize-1:-start-1]
    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))
    x_train = x[-trainSize-testSize-sampleSize:-testSize-sampleSize]
    y_train = y[-trainSize-testSize-sampleSize:-testSize-sampleSize]
    x_test = x[-testSize-sampleSize:-sampleSize]
    y_test = y[-testSize-sampleSize:-sampleSize]
    x_sample = x[-testSize-sampleSize-1:-1] 
    y_sample = y[-testSize-sampleSize-1:-1]
    print(len(data),len(x),len(y),len(x_sample),len(y_sample))
    return df,x_train,y_train,x_test,y_test,x_sample,y_sample

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

if __name__=='__main__':
    #gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    graph_dir = './graph/15-5/BRNN/'
    log_dir = './log/15-5/BRNN/'
    output_dir = './output/15-5/BRNN/'
    
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    trainSize = 10
    testSize = 5
    sampleSize = 5
    totalSize = trainSize+testSize+sampleSize
    for i in range(1):
        time.sleep(10)
        start = random.randint(0,3500)
        timeSequence = str(datetime.datetime.now())[20:26]
        df,x_train,y_train,x_test,y_test,x_sample,y_sample = dataLoad()
        x_actual,x_predict = birnn_orig.neuralNetwork.train(x_train,y_train,x_test,y_test,x_sample,y_sample)
        y_actual = y_sample[-sampleSize-1:-1]
        x_predict = x_predict[-sampleSize-1:-1]
        print('original data is: ')
        print(x_actual)
        print('prediction is: ')
        print(x_predict)
        f1score,accuracy,mse,mae = evaluation(y_actual,x_predict)
        print(f1score,accuracy,mse,mae )
        x_actual_ = np.squeeze(y_actual)
        x_predict_ = np.squeeze(x_predict)
        fig = visualize(x_actual_,x_predict_)  
        df_result = output(x_actual_,x_predict_)


