import pandas as pd
import numpy as np
import random
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ardl import ARDL,UECM
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.vecm import VECM

def arima(x_test):  
    order = (1,1,1)
    model = ARIMA(x_test,order=order)
    model = model.fit()
    print(model.summary())
    forcast_steps = 1
    x_predict = model.predict(start=len(x_test),end = len(x_test)+forcast_steps-1)
    model_name = 'ARIMA'

    return x_predict, model_name
def ar(x_test):  
    model = AutoReg(x_test,lags=1)
    model = model.fit()
    print(model.summary())
    forcast_steps = 1
    x_predict = model.predict(start=len(x_test),end = len(x_test)+forcast_steps-1)
    model_name = 'AR'

    return x_predict, model_name
def ardl(x_test):  
    model = ARDL(x_test,exog=None,lags=1,order=0)
    model = model.fit()
    print(model.summary())
    forcast_steps = 1
    x_predict = model.predict(start=len(x_test),end = len(x_test)+forcast_steps-1)
    model_name = 'ARDL'

    return x_predict, model_name
def uecm(x_train,y_train,x_test):  
    model = UECM(x_train,exog=y_train,lags=1,order=0)
    model = model.fit()
    print(model.summary())
    forcast_steps = 1
    x_predict = model.predict(start=len(x_test),end = len(x_test)+forcast_steps-1)
    model_name = 'UECM'

    return x_predict, model_name

def vecm(x_test):  
    model = VECM(x_test,k_ar_diff=1)
    model = model.fit()
    print(model.summary())
    forcast_steps = 1
    x_predict = model.predict(start=len(x_test),end = len(x_test)+forcast_steps-1)
    model_name = 'VECM'

    return x_predict, model_name


