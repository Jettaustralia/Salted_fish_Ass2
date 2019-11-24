import numpy as np
import pandas as pd
from xgboost import XGBClassifier
#import matplotlib.pyplot as plt
import pickle

# this function do not need to import,it will be called by next function.
def predict_one(s):  # 单个句子的预测函数
	list=[]
	for i in s:
		list.append(float(i))
#	print(list)
	array=np.array(list)
	array1 = array[np.newaxis,:]
#	print(array1)
	model = pickle.load(open("tmp.dat", "rb"))
	s_pred = model.predict(array1)
	return s_pred
	
''' 
import this function, and input a string with 'movie budget, movie popularity, movie release date, movie duration, movie type' like '28000000,150,2009-12-3,162,Adventure,Fantasy,Science Fiction,Animation,Family'
and this function will return a string which can be show to the users.
'''
	
def input_string(s):
	s = s.split(',')
	if len(s)<4:
		return 'incorrect input, Please tell me at least movie budget, movie popularity, movie release date, movie duration'
	if (str.isdigit(s[0])!=True and s[0]!='^[-+]?[0-9]+\.[0-9]+$'):
		return 'incorrect movie budget format'
	if (str.isdigit(s[1])!=True and s[1]!='^[-+]?[0-9]+\.[0-9]+$'):
		return 'incorrect movie popularity format'
	if (str.isdigit(s[3])!=True and s[3]!='^[-+]?[0-9]+\.[0-9]+$'):
		return 'incorrect movie duration format'
	for i in range(len(s[2])):
		if i ==4 or i==7:
			if s[2][i]!='-':
				return 'incorrect release date format1'
		else:
			if str.isdigit(s[2][i])!=True:
				return 'incorrect release date format'
	from datetime import datetime
	datetime_object = datetime.strptime(s[2], '%Y-%m-%d')
	day = datetime_object.timetuple().tm_yday
	year = s[2][:4]
	tmp = s[:2]+[day]+s[3:4]+[year]+[0]*20
	gener = ['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music', 'Documentary', 'Foreign','TV Movie']
	for i in s[4:]:
		if i in gener:
			tmp[gener.index(i)+5]=1		
#	print(tmp)
	R = int(predict_one(tmp).tolist()[0])
	if R == 2:
		print(R)
		return 'This will be a high-scoring movie, and in my opinion, his rating will be at least 6 points higher.'
	else:
		print(R)
		return 'Unfortunately, according to my predictions, it is difficult to get more than 6 points in the evaluation of this film.'

print(input_string('27000000,2,2015-02-13,109,Adventure,Romance,Action,Science Fiction'))

'''
a guide to let users make a right form input: 
If you need to make a prediction, enter the movie rating prediction and enter it in order: movie budget, movie popularity, movie release date, movie duration, movie type. With ',' as the interval, the input of the release date needs to be in the format of 'YYYY-MM-DD'.

for test(result should be 2)
28000000,150,2009-12-3,162,Adventure,Fantasy,Science Fiction,Animation,Family
'''
#if str.isdigit(s[0])!=True and s[0]!='^[-+]?[0-9]+\.[0-9]+$' and str.isdigit(s[1])!=True and s[1]!='^[-+]?[0-9]+\.[0-9]+$' and str.isdigit(s[3])!=True and s[3]!='^[-+]?[0-9]+\.[0-9]+$':
#		return 'incorrect input'