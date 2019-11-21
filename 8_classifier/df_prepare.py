

'''
one embedding测试
在GTX960上，36s一轮
经过30轮迭代，训练集准确率为95.95%，测试集准确率为89.55%
Dropout不能用太多，否则信息损失太严重
'''
from keras.models import load_model
import nltk
# nltk.download()
import numpy as np
import pandas as pd
import keras
# import jieba


# pos = pd.read_csv('pos_8_class.csv')
# neg = pd.read_csv('neg_8_class.csv')
#
# all_ = pos.append(neg, ignore_index=True)
# pd.to_numeric(all_['label'])
# # all_['label'].replace({1:0,2:0,3:1,4:1,7:2,8:2,9:3,10:3},inplace = True)
# all_['label'].replace({1:0,2:1,3:2,4:3,7:4,8:5,9:6,10:7},inplace = True)



all_ = pd.read_csv('whole_sentence.csv')
pd.to_numeric(all_['label'])





x1=all_.loc[all_['label']==0]
# print(x.to_string())
print(x1)
x2=all_.loc[all_['label']==1]
print(x2)
x3=all_.loc[all_['label']==2]
print(x3)
x4=all_.loc[all_['label']==3]
print(x4)
x5=all_.loc[all_['label']==4]
print(x5)
# x6=all_.loc[all_['label']==5]
# print(x6)
# x7=all_.loc[all_['label']==6]
# print(x7)
# x8=all_.loc[all_['label']==7]
# print(x8)
y = x1.append(x2, ignore_index=True)
y1 = y.append(x3, ignore_index=True)
y2 = y1.append(x4, ignore_index=True)
print(y2)