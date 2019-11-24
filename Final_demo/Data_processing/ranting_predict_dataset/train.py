#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Ranting_data.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])[:]
#print(data.shape)
rate = pd.read_csv('Ranting.csv')[:]
rate.rename(columns={'ranting':'0'},inplace = True)


##split at 6, can made dataset has about 50% 1 and 50% 2. 
rate.loc[rate['0']<=6.2,'0'] = 1
rate.loc[rate['0']>6.2,'0'] = 2
print(rate["0"].value_counts())

dataset = pd.concat([data,rate],axis=1)

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna() 

input_x = dataset.iloc[:, 1:-2].values
input_y = dataset.iloc[:, -1].values
#print(input_x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size = 0.2)


import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
 


# train model
model = xgb.XGBClassifier(min_child_weight =1,max_depth=4,learning_rate=0.01,n_estimators=800,objective = 'binary:logistic')
model.fit(x_train,y_train,eval_set=[(x_test, y_test)],eval_metric='error',verbose=1)

## for show the accurency of test dataset
y_pred = model.predict(x_test)
#print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
print('accuracy:%2.f%%'%(accuracy*100))
## model saved by save_model is too big.
#model.save_model('YYQ.model')
import pickle
pickle.dump(model, open("tmp.dat", "wb"))

# for load model
#model = pickle.load(open("right.dat", "rb"))


## for show the accurency of test dataset
#y_pred = model.predict(x_test)
#print(y_pred)
#accuracy = accuracy_score(y_test,y_pred)
#print('accuracy:%2.f%%'%(accuracy*100))

## show the model
#model.dump_model('dump.raw.txt')
#model.dump_model('dump.raw.txt','featmap.txt')

## show the inportance of different feature
plot_importance(model)
plt.show()

