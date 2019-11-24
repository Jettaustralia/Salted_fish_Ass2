# -*- coding:utf-8 -*
'''
5 class
'''

import nltk
import numpy as np
import pandas as pd
import keras
import json


filename_1 = "data_str.csv"
#filename_1 = "whole_sentence.csv"
filename_2 = "word_set.json"
filename_3 = "word_to_num_dict.json"
data_set = pd.read_csv('{}'.format(filename_1))
f_2 = open('{}'.format(filename_2), encoding='utf-8')
f_3 = open('{}'.format(filename_3), encoding='utf-8')
word_set = json.load(f_2)
word_to_num = json.load(f_3)


maxlen = 20 #截断词数
train_num = 120000

num_str = list(data_set['text_num_str'])
num_list = []
for i in num_str:
	i = list(i.split(','))
	i = [int(j) for j in i]
	num_list.append(i[:maxlen]) #!!!这里截断20个！！maxlen
	

x = np.array(num_list)# 句子数字序列列表
y = np.array(list(data_set['label']))# 句子标签列表
#print(y)
y = y.reshape((-1,1)) #调整标签形状（必须要的，把一个标签列表转成n句子个数的n个列表）
y = keras.utils.to_categorical(y, num_classes=5)


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

		
model = Sequential()
model.add(Embedding(len(word_to_num), 256, input_length=maxlen))
model.add(LSTM(output_dim=128, activation='tanh')) 
model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu', input_dim=maxlen))
model.add(Dense(5, activation='softmax'))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
							loss='categorical_crossentropy',
							metrics=['accuracy'])	
									
model.fit(x[:train_num], y[:train_num], validation_data = (x[train_num:], y[train_num:]) ,epochs=20, batch_size=128)			
model.save('model_20_epoch_15w.h5')

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
	sentence_num_list = [word_to_num[str(i)] for i in sen_words]
	return sentence_num_list


def predict_one(s): #单个句子的预测函数
	s = s.split(' ')
	s = np.array(text_to_number(s, maxlen,word_set,word_to_num))
	s = s.reshape((1, s.shape[0]))
	return model.predict_classes(s, verbose=0)
		
#while True:
#	s = input('sentence1: ')
#	print(predict_one(s))