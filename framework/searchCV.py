import pandas as pd
import numpy as np
import random
import sklearn
#from sentence_transformers import SentenceTransformer, util
#from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,ShuffleSplit,HalvingGridSearchCV

class searchCV():
    def gridSearchCV(model_base):
        param_dict = model_base.get_params()
        #param_dict = {'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        param_dict = {  'rbm__learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        print(param_dict)
        model_searchCV = GridSearchCV(model_base,param_dict)
        model_searchCV_name = 'gridSearchCV'
        return model_searchCV,model_searchCV_name
    def halvingGridSearchCV(model_base):
        #param_dict = model_base.get_params()
        param_dict = {  'rbm__learning_rate': [0.001,0.005,0.1],'par__epsilon': [0.05,0.1],  'par__tol': [0.001,0.01]}
        #param_dict ={'ccp_alpha': [0.0,0.1], 'min_impurity_decrease': [0.0,0.01], 'min_samples_leaf': [1,2],  'min_weight_fraction_leaf': [0.0,0.1]}
        print(param_dict)
        model_searchCV = HalvingGridSearchCV(model_base,param_dict,min_resources=2)
        model_searchCV_name = 'halvingGridSearchCV'
        return model_searchCV,model_searchCV_name