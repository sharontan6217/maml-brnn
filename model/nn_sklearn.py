import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, RANSACRegressor, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()



class myNN():
    def MLP():  
        #x_train = MinMaxScaler().fit_transform(x_train)
        #x_test = MinMaxScaler().fit_transform(x_test)
        model_base = MLPClassifier(random_state=0, verbose=True,alpha=0.001,solver='sgd',learning_rate_init=0.0001,max_iter=3000)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name='MLP'
        return model_base,model_base_name

    def BernoulliRBM(min_samples):  

        #x = np.concatenate((x_train,x_test),axis=0)
        #y = np.concatenate((y_train,y_test),axis=0)
        #x=scaler.fit_transform(x)
        #x_train=scaler.transform(x_train)   
        #x_test=scaler.transform(x_test)
        model_1 = BernoulliRBM(random_state=0, verbose=True, learning_rate=0.001, n_iter=200, n_components=2000)
        #model_2 = LogisticRegression(solver="liblinear", tol=0.001, C=6000)
        #model_2 = RidgeClassifier(tol=0.008, solver="auto",random_state=0)
        model_2 = DecisionTreeClassifier()
        #model_2 = SGDClassifier(alpha=2e-2, penalty="l2", loss="log_loss",random_state=0, verbose=True)
        #model_2 = LinearRegression()
        #model_2 = RANSACRegressor(min_samples=min_samples,random_state=0)
        #model_2 = PassiveAggressiveRegressor(random_state=0, verbose=True)
        #model_2 = IsotonicRegression()
        #model_2 = LinearSVR(tol=1e-4,random_state=0,  verbose=True)
        #model_2 = LinearSVC(C=10, tol=1e-4,dual=False, max_iter=1000,random_state=0, verbose=True)
        #model_2 = SGDRegressor(alpha=1e-2, penalty="l2", loss="squared_error",random_state=0, verbose=True)
        #model_2 =  NearestCentroid()
        model_base = Pipeline(steps=[("rbm", model_1 ), ("par", model_2 )])
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'BernoulliRBM'
        return model_base, model_base_name