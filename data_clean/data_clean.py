# -*- coding:utf-8 -*-
import nltk
import numpy as np
import pandas as pd
import json
import collections 

def get_raw_data(filename):
	'''
	Open the file to get the contents of the fourth column of the third-
	column of the source document and rename them
	Use nltk to divide sentences into words
	'''
	df = pd.read_csv('{}'.format(filename), sep='\t',usecols=[2,3],names=[0,'label'], header=0)
#	df.drop(['PhraseId'],axis=1,inplace=True)
#	df.drop(['SentenceId'],axis=1,inplace=True)
	df['words'] = df[0].apply(lambda a: list(nltk.word_tokenize(a)))
	return df

def get_content(df):
	content = []
	for i in df['words']:
		content.extend(i)
	return content

def get_word_set(content):
	'''
	Save all the words in the training set to the word_set file, 
	assign a unique digital representation to each different word, 
	and save it to the word_to_num_dict file.
	'''
	word_to_num = pd.Series(content).value_counts()	
	word_to_num[:] = list(range(1, len(word_to_num)+1))
	word_to_num[''] = 0 
	word_set = set(word_to_num.index)
	word_list = list(word_to_num.index)
	num_list = list(range(1, len(word_to_num)+1))
	word_to_num_dict = dict(map(lambda x,y:[x,y],word_list,num_list))
	return word_set, word_to_num, word_to_num_dict
	

def text_to_number(sentence, maxlen, word_set, word_to_num): 
	'''
	Convert all sentences in the training set to a list of numeric sequences 
	based on the word sequence number corresponding to each word
	'''
	sen_words = []
	for word in sentence:
		if word in word_set:
			sen_words.append(word)
	sen_words = sen_words[:maxlen] + ['']*max(0, maxlen-len(sen_words))
	return list(word_to_num[sen_words])

def save_csv(df, string):
	df = df.to_csv("{}.csv".format(string),index=0) 

def save_json(list_, string):
	with open('{}.json'.format(string),'w',encoding='utf-8') as f:
		f.write(json.dumps(list_, indent=4))


if __name__ == '__main__':
	'''
	maxlen = Maximum length of the sentence, the excess will be truncated
	'''
	maxlen = 100 
	filename = "train.tsv"
	df = get_raw_data(filename)
	content = get_content(df)
	word_set, word_to_num, word_to_num_dict = get_word_set(content)
#	print(word_set)
#	print(word_to_num_dict)
	word_set = list(word_set)
	save_json(word_set, "word_set")
	save_json(word_to_num_dict, "word_to_num_dict")
	df['text_num_list'] = df['words'].apply(lambda a: text_to_number(a, maxlen, word_set, word_to_num)) 
	idx = list(range(len(df)))
	np.random.shuffle(idx)
	df = df.loc[idx]
	df.drop(['words'],axis=1,inplace=True)
	save_csv(df, "data")
#	print(df)
	

		
	