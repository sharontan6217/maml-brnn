import pandas as pd
import numpy as np
import random
import sklearn
#from sentence_transformers import SentenceTransformer, util
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier


class selfTrainingClassifier():
    def selfTrainingClassifier(model_base):
        #print(y_mask)
        model_selfTraining = SelfTrainingClassifier(model_base,verbose=True)
        #model_classification = SelfTrainingClassifier(SGDClassifier(alpha=1e-5, penalty="l2", loss="log_loss"),verbose=True)
        #model_classification = SelfTrainingClassifier(LinearSVC(C=10, tol=1e-4,dual=False, max_iter=1000,random_state=0, verbose=True))
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #y_predict = model_classification.predict(x_test)
        model_selfTraining_name = 'selfTrainingClassifier'
        return model_selfTraining, model_selfTraining_name
    def LabelSpreading():

        model_selfTraining = LabelSpreading(kernel='rbf',alpha=0.9,tol=1e-4)
        #kernel{'knn', ‘rbf’}
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #y_predict = model_classification.predict(x_test)
        model_selfTraining_name = 'LabelSpreading'
        return model_selfTraining, model_selfTraining_name