# -*- coding:utf-8 -*-
'''
2 class
'''
from keras.models import load_model
import numpy as np
import pandas as pd
import nltk



maxlen = 300 #cut 
min_count = 20 

x=[]
with open('abc.txt', 'rt') as f:
	a = f.readline().replace('\n','')
	a = a.split(',')	


abc = pd.Series(x).value_counts()
abc = abc[abc >= min_count]


abc[:] = list(range(1, len(abc)+1))

abc[''] = 0 #padding with 0

print(abc[-1])


word_set=[]
with open('word_set.txt', 'rt') as f:
	for line in f:
		word_set .append(f.readline().replace('\n',''))
#print(word_set)
def doc2num(s, maxlen): 
#	s = s.split(' ')
	s = [i for i in s if i in word_set]
	s = s[:maxlen] + ['']*max(0, maxlen-len(s))
	return list(abc[s])



model = load_model('my_new_english_model_10_epoch.h5') #打开模型
def predict_one(s): #predict one sentence
	s = list(nltk.word_tokenize(s))
	s = np.array(doc2num(s, maxlen))
	s = s.reshape((1, s.shape[0]))
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
