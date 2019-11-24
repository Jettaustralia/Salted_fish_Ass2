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
	df = pd.read_excel('{}'.format(filename), header=None)
#	df.drop(['PhraseId'],axis=1,inplace=True)
#	df.drop(['SentenceId'],axis=1,inplace=True)
	df['words'] = df[0].apply(lambda a: list(nltk.word_tokenize(a)))
	return df
	
	
def get_whole_sentences(filename):
	df = pd.read_csv('{}'.format(filename), sep='\t',usecols=[1,2,3],names = ['SentenceId','Sentence','label'], header=0)
	col=df.iloc[:,0]
	arrs=col.values
	sentence_id = list(arrs)
	set_list = []
	index = 0
	for i in range(len(sentence_id)):
		if sentence_id[i] != index:
			set_list.append(i)
			index += 1
	return set_list
#	print(len(set_list))
#	print(set_list)
#	sentences = df.iloc[set_list]
#	print(sentences[:3])	
#	save_csv(sentences, "whole_sentence")
	
	
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

def list_to_str(list_):
	string = str(list_[0])
	for i in range(1,len(list_)):
		word = str(list_[i])
		string = string + ',' + word 
	return string 
	

if __name__ == '__main__':
	'''
	maxlen = Maximum length of the sentence, the excess will be truncated
	'''
	maxlen = 100 
	filename_1 = 'pos_english.xls'
	filename_2 = 'neg_english.xls'
	
	pos = get_raw_data(filename_1)
	neg = get_raw_data(filename_2)
	
	pos['label'] = 1
	neg['label'] = 0
	all_ = pos.append(neg, ignore_index=True)
	
	
	
	
#	get_whole_sentences(filename)
	df = get_raw_data(filename)
	content = get_content(df)
	word_set, word_to_num, word_to_num_dict = get_word_set(content)
#	print(word_set)
#	print(word_to_num_dict)
#	word_set = list(word_set)
#	save_json(word_set, "word_set")
#	save_json(word_to_num_dict, "word_to_num_dict")
#	df['text_num_list'] = df['words'].apply(lambda a: text_to_number(a, maxlen, word_set, word_to_num)) 
	text_num_list = df['words'].apply(lambda a: text_to_number(a, maxlen, word_set, word_to_num)) 
	str_ = []
	for each in text_num_list:
		tem = list_to_str(each)
		str_.append(tem)
	str_ = pd.Series(str_)
#	print(text_num_list[0])
	df['text_num_str'] = str_
	
#	idx = list(range(len(df)))
#	np.random.shuffle(idx)
#	df = df.loc[idx]
	df.drop(['words'],axis=1,inplace=True)
	sen_list = get_whole_sentences(filename)
	whole_sentences = df.iloc[sen_list]
#	print(whole_sentences)
	save_csv(whole_sentences, "whole_sentence")
	
#	save_csv(df, "data_str")
#	print(df)
	

		
	