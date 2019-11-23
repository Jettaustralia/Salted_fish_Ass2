## -*- coding: utf-8 -*-
#"""
#Created on Tue Jul 30 06:46:53 2019
#@author: rian-van-den-ander
#"""
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
##Loading data from preprocessed CSVs
#dataset_X_reimported = pd.read_csv('Encoded_X.csv')
#dataset_y_reimported = pd.read_csv('Encoded_y - revenue.csv')
#dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
#dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
#dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here
#
#X = dataset_reimported.iloc[:, 1:-2].values
#y = dataset_reimported.iloc[:, -1].values
#
## Splitting the dataset into the Training set and Test set
## I have a fairly large dataset of +- 4000 entries, so I'm going with 10% test data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
#
##This regressor was picked with gridsearch over many parameters - took 4 hours
#from xgboost import XGBRegressor
#regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
#                         subsample = 0.8, objective='reg:squarederror')
#regressor.fit(X, y)
#
#y_pred = regressor.predict(X_test)
#from sklearn.metrics import r2_score
#score = r2_score(y_test, y_pred)
#
#fig, ax = plt.subplots()
#ax.scatter(y_test, y_pred)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured revenue')
#ax.set_ylabel('Predicted revenue')
#plt.title('Measured versus predicted revenue')
#plt.ylim((50000000, 300000000))   # set the ylim to bottom, top
#plt.xlim(50000000, 300000000)     # set the ylim to bottom, top
#plt.show()




# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 06:46:53 2019
@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#,25,26,27,28,29,30,31,32,33,34
#Loading data from preprocessed CSVs ,445,991,961. ,usecols=[0,1,2,3,4,5,10,161,152,131,865,49,480,226,908,15,1219,632,348,932,587,415,65,129,18,158,16,1059,12,678]。 ,161,152,131,865,49,480,226,908,15,1219,632,348,932,587,415,65,129,18,158,16,1059,12,678





dataset_X_reimported = pd.read_csv('Ranting_data.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])[:]
print(dataset_X_reimported.shape)
dataset_y_reimported = pd.read_csv('Ranting.csv')[:]
dataset_y_reimported.rename(columns={'ranting':'0'},inplace = True)
#dataset_y_reimported = pd.to_numeric(dataset_y_reimported)

#dataset_X_reimported.apply(pd.to_numeric)
dataset_y_reimported.loc[dataset_y_reimported['0']<=6,'0'] = 1
#dataset_y_reimported.loc[dataset_y_reimported['0']>=6.8,'0'] = 3
dataset_y_reimported.loc[dataset_y_reimported['0']>6,'0'] = 2
print(dataset_y_reimported.to_string())
#dataset_y_reimported[dataset_y_reimported>=6]=1
#dataset_y_reimported[dataset_y_reimported<6]=0
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)

dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here

X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)







#This regressor was picked with gridsearch over many parameters - took 4 hours
from xgboost import XGBRegressor,XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
 


# 训练模型
model = xgb.XGBClassifier(min_child_weight =1,max_depth=4,learning_rate=0.01,n_estimators=500,objective = 'binary:logistic')
model.fit(X_train,y_train,eval_set=[(X_test, y_test),(X_train,y_train)],eval_metric='error',verbose=1)

#print(model.bst.best_iteration,model.bst.best_ntree_limit,model.bst.best_score)
## 对测试集进行预测
y_pred = model.predict(X_test)
print(y_pred)
#计算准确率
accuracy = accuracy_score(y_test,y_pred)
print('accuracy:%2.f%%'%(accuracy*100))

#model.save_model('YYQ.model')
import pickle
pickle.dump(model, open("YYQ.dat", "wb"))


#model = pickle.load(open("75.dat", "rb"))

#min_child_weight =1,max_depth=4,learning_rate=0.01,n_estimators=500,objective = 'binary:logistic'
#model = xgb.XGBClassifier()
#model.load_model("YYQ.model") 
##print(model.bst.best_iteration,model.bst.best_ntree_limit,model.bst.best_score)
### 对测试集进行预测
#y_pred = model.predict(X_test)
#print(y_pred)
##计算准确率
#accuracy = accuracy_score(y_test,y_pred)
#print('accuracy:%2.f%%'%(accuracy*100))

# 显示重要特征
#model.dump_model('dump.raw.txt')
# 导出模型和特征映射
#model.dump_model('dump.raw.txt','featmap.txt')

plot_importance(model)
plt.show()


def predict_one(s):  # 单个句子的预测函数
	s = s.split('\t')
	list=[]
	for i in s:
		list.append(float(i))
	array=np.array(list)
	array1 = array[np.newaxis,:]
	s_pred = model.predict(array1)
	return s_pred

while True:
#	c=[]
#	for i in [28000000.0,266,72.0,1993,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]:
#		c.append(int(i))
#	s = input('sentence1: ')
#	'28000000.0	266	72.0	1993	0	1	1	1	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0'
	s = '28000000.0'+'\t'+'150'+'\t'+'266'+'\t'+'72.0'+'\t'+'1993'+'\t'+'0'+'\t'+'1'+'\t'+'1'+'\t'+'1'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'1'+'\t'+'1'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'
	print(int(predict_one(s).tolist()[0]))
# 请输入电影评分预测后，按顺序输入：电影预算，电影受欢迎度，电影发布日期，电影时长，电影类型。
#['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary', 'Foreign','TV Movie']

#from datetime import datetime
#datetime_object = datetime.strptime(film_date, '%Y-%m-%d')
#X[l,4] = datetime_object.timetuple().tm_yday
#film_date like '2009-12-2'


#If you need to make a prediction, enter the movie rating prediction and enter it in order: movie budget, movie popularity, movie release date, movie duration, movie type. With ',' as the interval, the input of the release date needs to be in the format of 'YYYY-MM-DD'.



















#regressor = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
#model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')

#regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
#                         subsample = 0.8, objective='reg:squarederror')
#regressor.fit(X_train, y_train)
#
#y_pred = regressor.predict(X_test)
#ans = regressor.predict(X_test)
#
## 计算准确率
#cnt1 = 0
#cnt2 = 0
#for i in range(len(y_test)):
#	if ans[i] == y_test[i]:
#		cnt1 += 1
#	else:
#		cnt2 += 1
#
#print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
#
#
#plot_importance(regressor)
#plt.show()

#from sklearn.metrics import r2_score
#score = r2_score(y_test, y_pred)
#
#fig, ax = plt.subplots()
#ax.scatter(y_test, y_pred)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured revenue')
#ax.set_ylabel('Predicted revenue')
#plt.title('Measured versus predicted revenue')
#plt.ylim((50000000, 300000000))   # set the ylim to bottom, top
#plt.xlim(50000000, 300000000)     # set the ylim to bottom, top
#plt.show()