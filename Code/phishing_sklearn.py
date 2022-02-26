from warnings import simplefilter

import numpy as np
import pandas as pd
import sklearn
from numpy import genfromtxt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
#from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

simplefilter(action='ignore', category=FutureWarning)

###############################################################

feature=genfromtxt('phishing.csv',delimiter=',',usecols=(i for i in range(0,30)),skip_header=1)
target=genfromtxt('phishing.csv',delimiter=',',usecols=(-1),skip_header=1)
sc = StandardScaler()
sc.fit(feature)
target_label = LabelEncoder().fit_transform(target)
feature_std = sc.transform(feature)
test_size_val=0.20
x_train, x_test, y_train, y_test = train_test_split(feature_std, target_label, test_size=test_size_val, random_state=1)

print("Begin with test_size"+str(test_size_val)+":__________________________________")
###################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred.round())) 
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred.round()) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred.round())
    
    print('F1 score: %.2f' % f1_score(y_true=y_test,y_pred=y_pred.round()))
    print('Presicion: %.2f' % precision_score(y_true=y_test,y_pred=y_pred.round()))
    print('Recall: %.2f' % recall_score(y_true=y_test,y_pred=y_pred.round()))
    
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred.round(), rownames=['True'], colnames=['Predicted'], margins=True))



########################Logistic Regression##############################
print("########################Logistic Regression##############################")
clfLog = LogisticRegression()
clfLog.fit(x_train,y_train)
predictions = clfLog.predict(x_test)
print_stats_metrics(y_test, predictions)

########################Random Forest##############################
print("########################Random Forest##############################")
clfRandForest = RandomForestClassifier()
clfRandForest.fit(x_train,y_train)
predictions = clfRandForest.predict(x_test)
print_stats_metrics(y_test, predictions)
#######################Decision Tree#######################
print("#######################Decision Tree#######################")
clfDT = DecisionTreeRegressor()
clfDT.fit(x_train,y_train)
predictions = clfDT.predict(x_test)
print_stats_metrics(y_test, predictions)
#######################Naive Bayes#######################
print("#######################Naive Bayes#######################")
clfNB = GaussianNB()
clfNB.fit(x_train,y_train)
predictions = clfNB.predict(x_test)
print_stats_metrics(y_test, predictions)


