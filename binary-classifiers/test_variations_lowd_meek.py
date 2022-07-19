import os
import copy
import logging
from select import select

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from algorithms.OnlineBase import OnlineBase
from sklearn.linear_model import LogisticRegression
from lowd_meek import LordMeek
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dic={}

for data in ["diabetes", "income", "X_rated"]:
    dic[data]={}
    print("data")
    
    if data=="diabetes":
        X, y = load_svmlight_file('targets/targets/diabetes/test.scale', n_features=8)
        
        X = X.todense().tolist()
    elif data == "income":
        
        df = pd.read_csv("../data/adult.csv")
        df = pd.get_dummies(df)
        
        df["label"]= df['income_ >50K']
        
        #df.loc[df["label"]==0,"label"]= -1
        df=df.drop(['income_ <=50K', 'income_ >50K'], axis=1)
        
        X=df.drop("label",axis=1).values
        y=df["label"].values
        
    elif data == "X_rated":
        
        df = pd.read_csv("../data/GSShappiness.csv")
        
        df=df.dropna(subset="watched x-rated movies in the last year")
        df["label"]= df["watched x-rated movies in the last year"]=="YES"
        df= df.drop(["watched x-rated movies in the last year","id", "year"], axis=1)
        
        df= df.loc[~df["age"].isin(['89 OR OLDER',np.nan])]
        df= df.loc[~df["children"].isin(['EIGHT OR MORE',np.nan])]
        
        df[["age","children"]] =df[["age","children"]].astype(int)
        
        df= pd.get_dummies(df)
        
        
        X=df.drop("label",axis=1).values
        y=df["label"].values
        
        
    
    print(data)
    for model_type in ["LogisticRegression", "LinearSVC", "DecisionTree"]:
        print("--------------")
        print(model_type)
        
        dic[data][model_type]={}

        
        
        
        from sklearn.model_selection import train_test_split
        
        rest_x, X_test, rest_y, y_test = train_test_split(X,y,test_size=0.2,train_size=0.8, random_state=41)
        X_train_mod, X_train_steal, y_train_mod, y_train_steal = train_test_split(rest_x,rest_y,train_size =0.5, random_state=41)
            
        if model_type =="LogisticRegression":        
            clf = LogisticRegression()
        elif model_type =="LinearSVC":  
            clf = svm.LinearSVC()
        elif model_type == "DecisionTree":
            clf = DecisionTreeClassifier( random_state=42)
        clf.fit(X_train_mod, y_train_mod)
        
        
        # Perform Lowd-Meek
        mod = LordMeek(clf, (X_train_steal, y_train_steal),error=0.0001,  delta=1.0 / 10000)
        #mod.test()
        w, f = mod.find_continous_weights()
        
        b=mod.learn_intercept( X_train_steal, clf ,w,f)
        predictions= mod.predict(X_test, w, f, b)
        
        eval_df = pd.DataFrame([predictions,y_test,clf.predict(X_test)]).transpose()
        
        eval_df.columns = ["Prediciton", "Label", "Modeloutcome"]
        
        eval_df["Same_Predictions"]= eval_df["Prediciton"] == eval_df["Modeloutcome"]
        
        eval_df["Correct_Predictions"]= eval_df["Prediciton"] == eval_df["Label"]
        
        eval_df["Original_Predicitons"]= eval_df["Modeloutcome"] == eval_df["Label"]
        
        print(eval_df["Same_Predictions"].sum()/len(eval_df["Same_Predictions"]))
        
        print(eval_df["Correct_Predictions"].sum()/len(eval_df["Correct_Predictions"]))
        
        dic[data][model_type]["Same_Predictions"]=eval_df["Same_Predictions"].sum()/len(eval_df["Same_Predictions"])
        dic[data][model_type]["Accuracy_Stolen"]=eval_df["Correct_Predictions"].sum()/len(eval_df["Correct_Predictions"])
        dic[data][model_type]["Accuracy_Original"]=eval_df["Original_Predicitons"].sum()/len(eval_df["Original_Predicitons"])
        
    

    
