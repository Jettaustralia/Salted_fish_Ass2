# -*- coding: utf-8 -*-

import operator
from datetime import datetime
import json
import pandas as pd
import numpy as np


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

def delete_zero(list_,data_list):
	clean_list_1 = []
	clean_list_2 = []
	length = len(list_)
	for index in range(length):
		if list_[index] != 0:
			clean_list_1.append(list_[index])
			clean_list_2.append(data_list[index])
	return clean_list_1, clean_list_2

def process_date(data_list):
	column_id = 3
	year_list = []
	for i in range(len(data_list)):
		date = data_list[i,column_id]
		if len(date) == 10:
			year = int(date[:4])
#			print(year)
			day = datetime.strptime(date, '%Y-%m-%d')
			data_list[i,column_id] = day.timetuple().tm_yday
			year_list.append(year)
		else:
			data_list[i,column_id] = 0
			year_list.append(0)
#	data_list["year"] = year_list
	return data_list,year_list


def save_csv(df, string):
	df = df.to_csv("{}.csv".format(string),index=0) 	


if __name__ == '__main__':
	
	dataset = pd.read_csv('tmdb_5000_movies.csv')
	'''
	Some budgets are 0, since the size of dataset is not large
	Replace them with average budget
	'''
	ave_budget = dataset['budget'].mean()
	dataset['budget'] = dataset['budget'].replace(0,ave_budget)
	
	'''
	The value of movie rating is in colum 18
	'''
	rating_list = dataset.iloc[:, 18].values
#	print(rating_list)
	
	'''
	Choose the following features and process those data
	0: budget
	1: genres(the style of movie)
	8: popularity
	11: release_date
	13: runtime
	'''
	data_list = dataset.iloc[:, :].values
	data_list = data_list[:,[0,1,8,11,13]]
	'''
	Delete those movies with a rating of 0
	'''
	rating_list, data_list= delete_zero(rating_list,data_list)
#	rating_list = np.array(rating_list)
	data_list = np.array(data_list)
	'''
	Handling the release time of the movie
	Save as year, and date, date format is 365
	'''
#	print(data_list[0:1])
	data_list,year_list = process_date(data_list)
#	print(year_list)
	data_pd =  pd.DataFrame(data_list)
	data_pd.rename(columns = {0:"budget",1: "genres", 2:"popularity",3:"date",4:"runtime"},  inplace=True)
	data_pd["year"] = year_list
#	print(data_pd[0:1])
	'''
	Encode movie type
	'''
	data_pd = split_to_column(data_pd)
#	print(data_pd[0:1])
	save_csv(data_pd, 'Ranting_data.csv')
	rating_dict = {"ranting":rating_list}
	rating_pd = pd.DataFrame(rating_dict)
	save_csv(rating_pd, 'Ranting.csv')
	


	
	




