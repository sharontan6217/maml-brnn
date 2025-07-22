import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity, KDTree

import matplotlib.pyplot as plt


class myTree():
    def DecisionTree():  
        model_base = DecisionTreeClassifier()
        #model_tree=model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #y_predict = model_classification.predict(x_test)
        model_base_name = 'DecisionTreeClassifier'
        #plt.figure(figsize=(12,8))
        #sklearn.tree.plot_tree(model_tree,filled=True)
        #plt.show()
        return model_base, model_base_name
    def decisionRegressor():
        model_base = DecisionTreeRegressor()
        #x_train = np.array(x_train)
        #y_train = np.array(y_train)
        #x_test = np.array(x_test)

        #clf = clf.fit(x_train,y_train)
        #x_predict = clf.predict(x_test)
        model_base_name = 'DecisionTreeRegressor'
        #plt.figure(figsize=(12,8))
        #sklearn.tree.plot_tree(clf,filled=True)
        #plt.show()
        return model_base, model_base_name
    def RandomForest():  
        model_base =  RandomForestClassifier()
        #model_classification.fit(x_train, y_train)
        #score = model_classification.score(x_train,y_train)
        #print('score is: ',score)
        #y_predict = model_classification.predict(x_test)
        model_base_name = 'RandomForest'
        return model_base, model_base_name
