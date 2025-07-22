import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, RANSACRegressor, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

class myRegression():
    def LogisticRegression():
        model_base = LogisticRegression(random_state=0, verbose=True,solver="lbfgs", tol=0.008, C=6000)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'LogisticRegression'
        return model_base, model_base_name
    def LinearRegression():
        model_base = LinearRegression()
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'LinearRegression'
        return model_base, model_base_name
    def RidgeClassifier():
        model_base = RidgeClassifier(tol=0.02, solver="auto",random_state=0)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'RidgeClassifier'
        return model_base, model_base_name
    def RANSACRegressor(min_samples):
        model_base = RANSACRegressor(min_samples=min_samples,random_state=0)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'RANSACRegressor'

        return model_base, model_base_name
    def PassiveAggressiveRegressor():
        model_base = PassiveAggressiveRegressor(random_state=0, verbose=True)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'PassiveAggressiveRegressor'
        return model_base, model_base_name
    def IsotonicRegression():
        model_base = IsotonicRegression()
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'IsotonicRegression'
        return model_base, model_base_name
    def SGDRegressor():
        #x_train=scaler.fit_transform(x_train)
        #x_test=scaler.transform(x_test)
        model_base = SGDRegressor(alpha=1e-2, penalty="l2", loss="squared_error",random_state=0, verbose=True)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'SGDRegressor'
        return model_base, model_base_name
    def SGDClassifier():
        #x_train=scaler.fit_transform(x_train)
        #x_test=scaler.transform(x_test)
        model_base = SGDClassifier(alpha=2e-2, penalty="l2", loss="log_loss",random_state=0, verbose=True)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'SGDClassifier'
        return model_base, model_base_name
    def TheilSenRegressor():
        model_base = TheilSenRegressor(random_state=0, verbose=True,tol=0.001,max_iter=3000)
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #x_predict = model_classification.predict(x_test)
        model_base_name = 'TheilSenRegressor'
        return model_base, model_base_name