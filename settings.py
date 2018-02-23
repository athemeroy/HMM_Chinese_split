#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:58:56 2018

@author: zqun(athemeroy)
"""
import pickle

with open('bigram_trans_matrix.pickle', 'rb') as f:
#     Pickle the 'data' dictionary using the highest protocol available.
    bigram_trans_matrix = pickle.load(f)

#
with open('bigrams.pickle', 'rb') as f:
#     Pickle the 'data' dictionary using the highest protocol available.
    bigrams = pickle.load(f)

with open('emi_matrix.pickle', 'rb') as f:
#     Pickle the 'data' dictionary using the highest protocol available.
    emi_matrix = pickle.load(f)



with open('word2id.pickle', 'rb') as f:
    word2id = pickle.load(f)
    
    

with open('id2word.pickle', 'rb') as f:
    id2word = pickle.load(f)
    
with open('trigram_trans_matrix.pickle', 'rb') as f:
    trigram_trans_matrix = pickle.load(f)
    
with open('trigram_counter.pickle', 'rb') as f:
    trigram_counter = pickle.load(f)
