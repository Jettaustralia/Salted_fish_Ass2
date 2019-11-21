# # -*- coding:utf-8 -*-
#
# '''
# one embedding测试
# 在GTX960上，36s一轮
# 经过30轮迭代，训练集准确率为95.95%，测试集准确率为89.55%
# Dropout不能用太多，否则信息损失太严重
# '''
# from keras.models import load_model
# import numpy as np
# import pandas as pd
#
# pos = pd.read_excel('pos_english.xls', header=None)
# neg = pd.read_excel('neg_english.xls', header=None)
#
# #pos = pd.read_excel('pos_test_english.xls', header=None)
# #neg = pd.read_excel('neg_test_english.xls', header=None)
#
# #pos = pd.read_excel('test_pos.xls', header=None)
# #neg = pd.read_excel('test_neg.xls', header=None)
#
# pos['label'] = 1
# neg['label'] = 0
# all_ = pos.append(neg, ignore_index=True)
#
# maxlen = 300 #截断字数
# min_count = 20 #出现次数少于该值的字扔掉。这是最简单的降维方法
# #print("1111111111111")
# #print(all_[0])
#
# content = ''.join(all_[0])
# #print("2222222222222")
# #print(content)
# #abc = pd.Series(list(content)).value_counts()
#
# abc = pd.Series(content.split(' ')).value_counts()
# abc = abc[abc >= min_count]
#
# #print("333333333333")
# #print(abc)
#
# abc[:] = list(range(1, len(abc)+1))
#
# #print("44444444444")
# #print(abc)
#
# abc[''] = 0 #添加空字符串用来补全
# word_set = set(abc.index)
# print(word_set)
# #print("555555555555")
# #print(word_set)
#
#
#
# def doc2num(s, maxlen):
# 		s = [i for i in s if i in word_set]
# 		s = s[:maxlen] + ['']*max(0, maxlen-len(s))
# #		print(s)
# 		return list(abc[s])
#
#
#
# all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))
# #print(all_[0])
# #手动打乱数据
# idx = list(range(len(all_)))
# np.random.shuffle(idx)
# all_ = all_.loc[idx]
#
# #按keras的输入要求来生成数据
# #x = np.array(list(all_['doc2num']))
# #y = np.array(list(all_['label']))
# #y = y.reshape((-1,1)) #调整标签形状
# #
# #from keras.models import Sequential
# #from keras.layers import Dense, Activation, Dropout, Embedding
# #from keras.layers import LSTM
# #
# ##建立模型
# #model = Sequential()
# #model.add(Embedding(len(abc), 256, input_length=maxlen))
# #model.add(LSTM(128))
# #model.add(Dropout(0.5))
# #model.add(Dense(1))
# #model.add(Activation('sigmoid'))
# #model.compile(loss='binary_crossentropy',
# #							optimizer='adam',
# #							metrics=['accuracy'])
# #
# #batch_size = 128
# #train_num = 15000
# #
# #model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=60)
# #
# #model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
# #
# #
# #model.save('my_english_model.h5') #保存为h5模型
#
#
#
# model = load_model('my_english_model.h5') #打开模型
# def predict_one(s): #单个句子的预测函数
# 	s = s.split(' ')
# 	s = np.array(doc2num(s, maxlen))
# 	s = s.reshape((1, s.shape[0]))
# #	print(model.predict_classes(s, verbose=0))
# 	return model.predict_classes(s, verbose=0)[0][0]
# while True:
# 	s = input('sentence1: ')
# 	print(predict_one(s))


# -*- coding:utf-8 -*-

'''
one embedding测试
在GTX960上，36s一轮
经过30轮迭代，训练集准确率为95.95%，测试集准确率为89.55%
Dropout不能用太多，否则信息损失太严重
'''
from keras.models import load_model
import numpy as np
import pandas as pd
# import jieba

pos = pd.read_excel('pos_english.xls', header=None)
neg = pd.read_excel('neg_english.xls', header=None)

# pos = pd.read_excel('pos_test_english.xls', header=None)
# neg = pd.read_excel('neg_test_english.xls', header=None)

# pos = pd.read_excel('test_pos.xls', header=None)
# neg = pd.read_excel('test_neg.xls', header=None)

pos['label'] = 1
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)

all_['words'] = all_[0].apply(lambda s: s.split(' '))

maxlen = 300  # 截断字数
min_count = 20  # 出现次数少于该值的字扔掉。这是最简单的降维方法
# print("1111111111111")
# print(all_[0])

# content = ''.join(all_[0])
content = []
for i in all_['words']:
	content.extend(i)

# print("2222222222222")
# print(content)
# abc = pd.Series(list(content)).value_counts()

# abc = pd.Series(content.split(' ')).value_counts()
abc = pd.Series(content).value_counts()
print(abc)
abc = abc[abc >= min_count]

# print("333333333333")
# print(abc)

abc[:] = list(range(1, len(abc) + 1))

# print("44444444444")
# print(abc)

abc[''] = 0  # 添加空字符串用来补全
word_set = set(abc.index)


# print("555555555555")
# print(word_set)


def doc2num(s, maxlen):
	# print(s)
	#	s = s.split(' ')
	s = [i for i in s if i in word_set]
	s = s[:maxlen] + [''] * max(0, maxlen - len(s))
	return list(abc[s])


# all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))
all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))
# print(all_[0])
# 手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

# 按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM

#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
							optimizer='adam',
							metrics=['accuracy'])

batch_size = 128
train_num = 15000

model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=10)

model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)


model.save('my_new_english_model.h5') #保存为h5模型


# model = load_model('my_english_model_1.h5')  # 打开模型


def predict_one(s):  # 单个句子的预测函数
	s = s.split(' ')
	s = np.array(doc2num(s, maxlen))
	# print(s)
	s = s.reshape((1, s.shape[0]))
	return model.predict_classes(s, verbose=0)[0][0]


while True:
	s = input('sentence1: ')
	print(predict_one(s))