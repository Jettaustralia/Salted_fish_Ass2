# -*- coding:utf-8 -*-
'''
2 class
'''
from keras.models import load_model
import numpy as np
import pandas as pd
import json


 #打开模型
def predict_one(s): #单个句子的预测函数
	
	maxlen = 300 #cut
	min_count = 20 
	pos = pd.read_excel('pos_english.xls', header=None)
	neg = pd.read_excel('neg_english.xls', header=None)

	pos['label'] = 1
	neg['label'] = 0
	all_ = pos.append(neg, ignore_index=True)

	all_['words'] = all_[0].apply(lambda s: s.split(' ')) 
	content = []
	for i in all_['words']:
		content.extend(i)

	abc = pd.Series(content).value_counts()
	abc = abc[abc >= min_count]


	abc[:] = list(range(1, len(abc)+1))

	abc[''] = 0 #padding with 0
	word_set = set(abc.index)
	
	def doc2num(s, maxlen): 
	#	s = s.split(' ')
		s = [i for i in s if i in word_set]
		print(s)
		s = s[:maxlen] + ['']*max(0, maxlen-len(s))
		sentence_num_list = [word_to_num[str(i)] for i in s]
#		return list(abc[s])
		return sentence_num_list
		
	s = s.split(' ')
	s = np.array(doc2num(s, maxlen))
	s = s.reshape((1, s.shape[0]))
	model = load_model('my_new_english_model_10_epoch.h5')
	return model.predict_classes(s, verbose=0)[0][0]
## please call this function derictly
## argument is the input from users
## ourput wii be the reply	
def input_sentence(s):
	R = predict_one(s)
	if R == 0:
		return 'Your review is a negtive review'
	else:
		return 'Your review is a positive review'
#print(input_sentence('this is a bad movie, just say no to this movie'))


