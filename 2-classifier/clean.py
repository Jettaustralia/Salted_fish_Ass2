from keras.models import load_model
import numpy as np
import pandas as pd

pos = pd.read_excel('pos_english.xls', header=None)
neg = pd.read_excel('neg_english.xls', header=None)

pos['label'] = 1
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)

all_['words'] = all_[0].apply(lambda s: s.split(' ')) 


maxlen = 300 #截断字数
min_count = 20 #出现次数少于该值的字扔掉。这是最简单的降维方法

content = []
for i in all_['words']:
	content.extend(i.replace('\n','').replace(',','').replace('/>','').replace('<br',''))
	
file = open('abc.txt', 'w')
for i in content:
	file.write(i)
	file.write(',')
file.close()

print(content)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]


abc[:] = list(range(1, len(abc)+1))


abc[''] = 0 #添加空字符串用来补全
abc.to_csv("abc.csv")
print(abc)
print(type(abc))

word_set = set(abc.index)
word_set = list(word_set)
#print()
print(len(word_set))
file = open('word_set.txt', 'w')
for i in word_set:
	file.write(i)
	file.write('\n')
file.close()
print("555555555555")
print(word_set)




