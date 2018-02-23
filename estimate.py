#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:24:30 2018

@author: zqun(athemeroy)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:07:44 2018

@author: zqun(athemeroy)
"""
from collections import Counter, OrderedDict
import re
import pickle


with open('tagged_trainCorpus.utf8') as f:
    tagged_train_str = f.read()
import numpy as np

uni_word = set(re.sub('/[SBME] ', '', tagged_train_str))
print('finding tags...')
tags = re.findall('[SBME]',tagged_train_str)




print('finding and counting trigrams...')
trigram_counter = Counter([tags[i]+tags[i+1]+tags[i+2] for i in range(len(tags)-3)])
print('counting bigrams...')
bigrams = {}
for (i,j) in trigram_counter.items():
    if bigrams.get(i[:2]):
        bigrams[i[:2]] += j
    else:
        bigrams[i[:2]] = j
print('counting unigrams')
unigrams = {}
for (i,j) in bigrams.items():
    if unigrams.get(i[0]):
        unigrams[i[0]] += j
    else:
        unigrams[i[0]] = j

print('constructing trigram transfer matrix')
trigram_trans_matrix = np.zeros((16,4))
order = 'SBME'
bi_order = []
for i in order:
    for j in order:
        bi_order.append(i+j)
        
for l, left in enumerate(bi_order):
    for r,right in enumerate(order):
        if bigrams.get(left) and trigram_counter.get(left+right):
            trigram_trans_matrix[l][r] = trigram_counter.get(left+right)/bigrams.get(left)
            
print('constructing bigram transfer matrix')
bigram_trans_matrix = np.zeros((4,4))
for l, left in enumerate(order):
    for r,right in enumerate(order):
        if bigrams.get(left+right):
            bigram_trans_matrix[l][r] = bigrams.get(left+right)/unigrams.get(left)
            
print('splitting string...')
word_tag_tuple_list = [(k.split('/')[0],k.split('/')[1]) for k in tagged_train_str.split(' ')]
del tagged_train_str
print('counting word-tag tuples')
word_tag_tuple_od = OrderedDict(Counter(word_tag_tuple_list))


print('constructing emission matrix')
order = 'SBME'
emission_matrix = np.zeros((len(uni_word)+1, 4)) # 最后一行是≈，代表不认识的字

word2id = {i:v for i,v in zip(uni_word,range(len(uni_word)))}
word2id['≈'] = len(word2id)
id2word = {v:i for i,v in word2id.items()}
for word, wid in word2id.items():
    for t,tag in enumerate(order):
        if word_tag_tuple_od.get((word,tag)):
            emission_matrix[wid][t] = (word_tag_tuple_od[(word,tag)]-0.5)/unigrams.get(tag)



import pandas as pd
emi = pd.DataFrame(emission_matrix)

emi_matrix = np.zeros((len(uni_word)+1, 4)) 
def fillin(column, num):
    summation = column.sum()
    remainer = 1 - summation
    avg = remainer / len(column)
    for index, value in enumerate(column):
        emi_matrix[index][num] = value+avg

for i in range(4):
    fillin(emi[i],i)

with open('word2id.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(word2id, f, pickle.HIGHEST_PROTOCOL)
    
with open('id2word.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(id2word, f, pickle.HIGHEST_PROTOCOL)

with open('emi_matrix.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(emi_matrix, f, pickle.HIGHEST_PROTOCOL)


with open('unigrams.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(unigrams, f, pickle.HIGHEST_PROTOCOL)


with open('bigram_trans_matrix.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(bigram_trans_matrix, f, pickle.HIGHEST_PROTOCOL)


with open('bigrams.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(bigrams, f, pickle.HIGHEST_PROTOCOL)


with open('trigram_counter.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(trigram_counter, f, pickle.HIGHEST_PROTOCOL)

with open('trigram_trans_matrix.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(trigram_trans_matrix, f, pickle.HIGHEST_PROTOCOL)
