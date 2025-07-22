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
    start = random.randint(totalSize,len(df_imts)-totalSize)
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
def selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_test,y_test):
    training_model = opt.selftraining_name
    if training_model.lower() == 'selfTrainingClassifier'.lower():
        y_predict,model_selfTraining,model_selfTraining_name  = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train_mask,x_test,y_test)
    else:
       y_predict,model_selfTraining,model_selfTraining_name = selfTrainingClassifier.LabelSpreading(model_predict,x_train,y_train,x_test,y_test)

    return y_predict

def meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt):
    gc.collect()
    meta_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    meta_name = opt.metalearning_name
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    # Dummy tasks for demonstration
    for j in range(predictSize):
        if opt.with_selftraining == True:
            if j==0:
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
                    maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.001, meta_steps=400, inner_steps=5)
                #y_predict = model_predict.predict(x_test)
                y_predict=selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_test,y_test)
            else:
                if meta_name.lower()=='reptile'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train, 'Y':  y_train},
                    {'X': x_actual, 'Y':  y_predict}]
                    training,testing,model_predict=reptile.reptile_train(model_predict, tasks, meta_optimizer, meta_steps=600)
                    reptile.reptile_visualize(training,testing,timeSequence,start,graph_dir)
                elif meta_name.lower()=='MAML'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train[:testSize], 'Y':  y_train[:testSize]},
                    {'X': x_train[-testSize-1:-1], 'Y':  y_train[:testSize]},
                    {'X': x_actual, 'Y':  y_predict}]
                    maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.001, meta_steps=400, inner_steps=5)
                y_predict=selfTrainingModel(model_predict,opt,x_train,y_train_mask,y_train,x_actual,y_predict)
                #y_predict = model_predict.predict(x_actual)
            x_new=y_predict[-1].reshape(-1,1)
            x_actual1 = x[trainSize+j:trainSize+testSize+j-1]
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual=np.reshape(x_actual,(len(x_actual),1,1))
            #x_new=x_predict[-1].reshape(-1,1)
            #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
            #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
            #y_predict.append(x_new)
        else:
            if j==0:
                if meta_name.lower()=='reptile'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train, 'Y':  y_train},
                    {'X': x_test, 'Y':  y_test}]
                    training,testing,model_predict=reptile.reptile_train(model_predict, tasks, meta_optimizer, meta_steps=400)
                    reptile.reptile_visualize(training,testing,timeSequence,start,graph_dir)
                elif meta_name.lower()=='MAML'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train[:testSize], 'Y':  y_train[:testSize]},
                    {'X': x_train[-testSize-1:-1], 'Y':  y_train[:testSize]},
                    {'X': x_test, 'Y':  y_test}]
                    maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.01, meta_steps=400, inner_steps=5)
                y_predict = model_predict.predict(x_test)

            else:
                if meta_name.lower()=='reptile'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train, 'Y':  y_train},
                    {'X': x_actual, 'Y':  y_predict}]
                    training,testing,model_predict=reptile.reptile_train(model_predict, tasks, meta_optimizer, meta_steps=400)
                    reptile.reptile_visualize(training,testing,timeSequence,start,graph_dir)
                elif meta_name.lower()=='MAML'.lower():
                    # Train the model using reptile
                    tasks = [
                    {'X': x_train[:testSize], 'Y':  y_train[:testSize]},
                    {'X': x_train[-testSize-1:-1], 'Y':  y_train[:testSize]},
                    {'X': x_actual, 'Y':  y_predict}]
                    maml.maml_train(model_predict, tasks, meta_optimizer, inner_lr=0.01, meta_steps=400, inner_steps=5)
                y_predict = model_predict.predict(x_actual)
            x_new=y_predict[-1].reshape(-1,1)
            x_actual1 = x[trainSize+j:trainSize+testSize+j-1]
            x_actual = np.append(x_actual1,x_new,axis=0)
            x_actual=scaler.transform(x_actual)
            x_actual=np.reshape(x_actual,(len(x_actual),1,1))
            #x_new=x_predict[-1].reshape(-1,1)
            #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
            #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
            #y_predict.append(x_new)
        j=j+1
    y_predict = y_predict[-predictSize-1:-1]
    y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    print('start is: ',start)
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
            x_actual1 = x[trainSize+j:trainSize+testSize+j-1]
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
            x_actual1 = x[trainSize+j:trainSize+testSize+j-1]
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
                    y_estimate_ = np.mean([y_train_mask[i-1],y_train_mask[i+1]])
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
    data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    graph_dir = './graph/100-20/'
    log_dir = './log/100-20/'
    output_dir = './output/100-20/'

    batch_size=32
    epochs=3000
    drop_out=0.2
    patience=5
    gru_units=20
    dense_units=5
    input_shape=(None,1)
        
    
    poolSize = 1080
    trainSize = 100
    testSize = 20
    predictSize = 5
    totalSize = trainSize+testSize+predictSize
    opt = get_parser()
    df,orig = dataLoad()
  
    model_predict = neuralNetwork.myBiRNN(gru_units=gru_units,drop_out=drop_out,input_shape=input_shape)
    if opt.with_metalearning==True:
        model_metalearning_name = opt.metalearning_name
        if opt.with_selftraining == True:
            model_selfTraining_name = opt.selftraining_name
            log_dir = log_dir+model_metalearning_name+'/'+model_selfTraining_name+'/BRNN/'
            output_dir = output_dir+model_metalearning_name+'/'+model_selfTraining_name+'/BRNN/'
            graph_dir = graph_dir+model_metalearning_name+'/'+model_selfTraining_name+'/BRNN/'
        else:
            log_dir = log_dir+model_metalearning_name+'/BRNN/'
            output_dir = output_dir+model_metalearning_name+'/BRNN/'
            graph_dir = graph_dir+model_metalearning_name+'/BRNN/'
    else:
        if opt.with_selftraining == True:
            model_selfTraining_name = opt.selftraining_name
            log_dir = log_dir+model_selfTraining_name+'/BRNN/'
            output_dir = output_dir+model_selfTraining_name+'/BRNN/'
            graph_dir = graph_dir+model_selfTraining_name+'/BRNN/'
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
        gc.collect()
        timeSequence = str(datetime.datetime.now())[20:26]
        x,y,x_train,y_train,x_test,y_test,y_actual = dataSplit(orig,trainSize,testSize,predictSize)
        y_train_mask = datamask(y_train)
        if opt.with_metalearning==True:
            y_predict, y_actual, model_predict = meta_train(model_predict,x_train,y_train,x_test,y_test,y_actual,opt)
        else:
            y_predict, y_actual, model_predict = train(model_predict,x_train,y_train,x_test,y_test,y_actual)

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


