# -*- coding: utf-8 -*-
'''
Processing on 'genres' column and translate it into 1 and 0
Compute all the features of the movie
If the movie contain the feature, the feature column will be 1
Else will be 0
'''

import operator
import json
import pandas as pd

def check(list_):
	try:
		flag = json.loads(list_)
	except:
		return False
	return True


def apperance_num(data,col_name):
	all_classes = {}
	for each in data:
		if(check(each[1])): 
			for i in json.loads(each[1]):
				id_ = i[col_name]                
				if id_ in all_classes:
					all_classes[id_] += 1                   
				else:
					all_classes[id_] = 1
	return all_classes

def trans_to_list(all_classes):
	list_ = sorted(all_classes.items(), key=operator.itemgetter(1), reverse=True) 
	return dict(list_)


def split_to_column(data):
	col_name = "name"
	data_list = data.iloc[:, :].values

	all_classes = apperance_num(data_list,col_name)
	freq_dict = trans_to_list(all_classes)
	
	check_list = []
	row_id = 0

	for row in data_list:        
		if(check(row[1])):       
			for i in json.loads(row[1]):                
				index = i[col_name]                                
				if index not in check_list:
					check_list.append(index)
					data[index]=0
				data.loc[row_id,index] = 1                       
			row_id += 1
		else:
			data.drop(data.index[row_id])
			
	data = data.drop(data.columns[1], 1)    
	return data
	


