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
    def LogisticRegression(x_train, y_train,x_test):
        model_classification = LogisticRegression(random_state=0, verbose=True,solver="sgd", tol=2, C=6000)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'LogisticRegression'
        return x_predict, model_name
    def LinearRegression(x_train, y_train,x_test):
        model_classification = LinearRegression()
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'LinearRegression'
        return x_predict, model_name
    def RidgeClassifier(x_train, y_train,x_test):
        model_classification = RidgeClassifier(tol=0.0008, solver="auto",random_state=0)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'RidgeClassifier'
        return x_predict, model_name
    def RANSACRegressor(x_train, y_train,x_test):
        model_classification = RANSACRegressor(min_samples=len(x_train),random_state=0)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'RANSACRegressor'

        return x_predict, model_name
    def PassiveAggressiveRegressor(x_train, y_train,x_test):
        model_classification = PassiveAggressiveRegressor(random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'PassiveAggressiveRegressor'
        return x_predict, model_name
    def SGDRegressor(x_train, y_train,x_test):
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model_classification = SGDRegressor(alpha=1e-2, penalty="l2", loss="squared_error",random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'SGDRegressor'
        return x_predict, model_name
    def SGDClassifier(x_train, y_train,x_test):
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model_classification = SGDClassifier(alpha=2e-2, penalty="l2", loss="log_loss",random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'SGDClassifier'
        return x_predict, model_name
    def TheilSenRegressor(x_train, y_train,x_test):
        model_classification = TheilSenRegressor(random_state=0, verbose=True)
        model_classification.fit(x_train, y_train)
        score = model_classification.score(x_train,y_train)
        print('score is: ',score)
        x_predict = model_classification.predict(x_test)
        model_name = 'TheilSenRegressor'
        return x_predict, model_name