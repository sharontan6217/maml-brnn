import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import random
import model
from model import  tsa
import datetime
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
    x = data[-poolSize-1:-2]
    y = data[-poolSize:-1]
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    print(len(data),len(x),len(y))
    return df,x,y
def train(x,y,start,trainSize,testSize,predictSize):
    #x_train = x[-start:-start+trainSize]
    #y_train = y[-start:-start+trainSize]
    #x_test = x[-start:-start+trainSize+testSize]
    #y_test = y[-start:-start+trainSize+testSize]
    #y_actual = y[-start+trainSize+testSize:-start+trainSize+testSize+predictSize]
    #x_actual = x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)
    #model,model_name=tsa.arima(x_test)
    total = []
    for n in range(trainSize+testSize+predictSize+1):
        number = random.randint(0,len(x))
        total.append(number)
    total = np.array(total).sort()
    print(total)
    x_total = total[:-2]
    y_total = total[1:]
    print(len(x_total))
    print(len(y_total))
    x_train = x_total[-start:-start+trainSize]
    y_train = y_total[-start:-start+trainSize]
    x_test = x_total[-start+trainSize:-start+trainSize+testSize]
    y_test = y_total[-start+trainSize:-start+trainSize+testSize]
    y_actual = y_total[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize]

    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    y_predict=[]
    for j in range(predictSize):
        if j==0:
            x_predict, model_name = tsa.arima(x_test)
            #x_predict, model_name = tsa.ar(x_test)
            #x_predict, model_name = tsa.uecm(x_train,y_train,x_test)
            #x_predict, model_name = tsa.ardl(x_test)
        else:
            x_predict, model_name = tsa.arima(x_actual) 
            #x_predict, model_name = tsa.ar(x_actual) 
            #x_predict, model_name = tsa.uecm(x_train,y_train,x_actual) 
            #x_predict, model_name = tsa.ardl(x_actual) 
        x_new=x_predict.reshape(-1,1)
        x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
        y_predict.append(x_new)
        j=j+1
    y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    print('start is: ',start)
    return y_predict, y_actual, model_name

def evaluation(actual,predict):
    f1score = f1_score(actual.astype('int32'),predict.astype('int32'),average='micro')
    accuracy = accuracy_score(actual.astype('int32'),predict.astype('int32'))
    mse = mean_squared_error(actual,predict)
    mae = mean_absolute_error(actual,predict)
    with open (log_dir+model_name+'/'+'result_'+model_name+'.log','a') as f:
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
    fig_name='test_scenario_'+model_name+'_'+timeSequence+'_'+str(start)+'.png'
    plt.savefig(graph_dir+model_name+'/'+fig_name)
    plt.close()      

    return fig
def output(actual,predict):
    df_result=pd.DataFrame()
    df_result['actual']=actual
    df_result['predict']=predict
    output_name = 'output_'+model_name+'_'+timeSequence+'_'+str(start)+'.csv'
    df_result.to_csv(output_dir+model_name+'/'+output_name)
    return df_result

if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    graph_dir = './graph/15-5/'
    log_dir = './log/15-5/'
    output_dir = './output/15-5/'
    
    poolSize = 3600
    trainSize = 100
    testSize = 15
    predictSize = 5
    totalSize = trainSize+testSize+predictSize

    df,x,y = dataLoad()
    for i in range(50):
        timeSequence = str(datetime.datetime.now())[20:26]
        start = random.randint(totalSize,poolSize-1)
        #sampleSize = 15
        y_predict,y_actual, model_name = train(x,y,start,trainSize,testSize,predictSize)
        if os.path.exists(graph_dir+model_name+'/')==False:
            os.mkdir(graph_dir+model_name+'/')
        if os.path.exists(log_dir+model_name+'/')==False:
            os.mkdir(log_dir+model_name+'/')
        if os.path.exists(output_dir+model_name+'/')==False:
            os.mkdir(output_dir+model_name+'/')
        f1score,accuracy,mse,mae = evaluation( y_actual,y_predict)
        print(f1score,accuracy,mse,mae )
        x_actual_ = np.squeeze( y_actual)
        x_predict_ = np.squeeze(y_predict)
        fig = visualize(x_actual_,x_predict_)  

        df_result = output(x_actual_,x_predict_)
        i+=1

