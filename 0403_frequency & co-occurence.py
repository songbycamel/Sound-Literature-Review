# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 06:49:10 2023

@author: YEIN

reference: https://foreverhappiness.tistory.com/38
 
"""

import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from itertools import combinations
from collections import Counter
import numpy as np

# import csv data

f=open('0421_EV.csv','r',encoding='utf-8-sig')
rdr=csv.reader(f)
data=[]
for line in rdr:
  temp=[]
  for i in line:
    if len(i)==0:
      line.remove(i)
  temp=' '.join(line)
  data.append(temp)


f.close()

vocab = set(word_tokenize(' '.join(data)))
print('Vocabulary:\n',vocab,'\n')
token_sent_list = [word_tokenize(sen) for sen in data]
print('Each sentence in token form:\n',token_sent_list,'\n')



# Frequency analysis

def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result

for_freq = flatten(token_sent_list)
c = Counter(for_freq)
print(c.most_common(10))


# co-occurence matrix

co_occ = {ii:Counter({jj:0 for jj in vocab if jj!=ii}) for ii in vocab}
k=2

for sen in token_sent_list:
    for ii in range(len(sen)):
        if ii < k:
            c = Counter(sen[0:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        elif ii > len(sen)-(k+1):
            c = Counter(sen[ii-k::])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        else:
            c = Counter(sen[ii-k:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c

# Having final matrix in dict form lets you convert it to different python data structures
co_occ = {ii:dict(co_occ[ii]) for ii in vocab}

# display(co_occ)

co_occ = pd.DataFrame(co_occ)

co_occ.to_csv("co-occurence matrix_0421EV.csv", encoding='utf-8-sig')