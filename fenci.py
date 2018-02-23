#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:06:14 2018

@author: zqun(athemeroy)
"""

from settings import word2id, emi_matrix
from settings import bigram_trans_matrix,trigram_trans_matrix

import re

#from collections import OrderedDict, Counter
import numpy as np
#from collections import deque

def viterbi(st, mode=0, alpha=0.8):
    """
    st:要分词的字符串
    mode:
        0：混合
        2：2-grams
        3：3-grams
    alpha:仅mode=0使用：3-grams所占的比重，range(0,1)
    """
    #假设第一个字的隐状态[0.5, 0, 0, 0.5]
    def preprocess(s):
        pre_s = re.split('[; |,。，.]',s)
        return pre_s
    real_result = ''
    for string in preprocess(st):
        length = len(string)
        raw_string = string
        string += "Ω"
        for i in string:
            if not word2id.get(i):
                string = string.replace(i,"≈")
    #    print(string)
        # graph是一个双重矩阵，一重表示到某节点的概率
        # 二重表示到底是哪个上层节点过来的
        graph = np.zeros((4, length+2 ,2))
        init = np.array([0.5, 0.5, 0, 0])
        graph[:,0,0] = init
        
        for index, char in enumerate(string):
            if index == 0:
                graph[:,0,0] = graph[:,0,0] * emi_matrix[word2id[char]]
            else:
                if mode == 2: # 使用bigram模型，一阶马尔科夫
                    graph[:,index, 1] = [np.argmax(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i]) for i in range(4)]
                    graph[:,index, 0] = [np.max(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
                elif mode == 3: # 使用trigram模型，二阶马尔科夫
                    if index == 1:
                        graph[:,index,0] = [np.max(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
                        graph[:,index,1] = [np.argmax(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i]) for i in range(4)]
                    else:
                        matrix_index = [graph[j,index-1,1]*4+j for j in range(4)]
                        matrix_index = [int(k) for k in matrix_index]
                        graph[:,index, 0] = [np.max(graph[:,index-1,0]*trigram_trans_matrix[matrix_index,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
                        graph[:,index, 1] = [np.argmax(graph[:,index-1,0]*trigram_trans_matrix[matrix_index,i]*emi_matrix[word2id[char],i]) for i in range(4)]
                elif mode == 0:
                    if index == 1:
                        graph[:,index,0] = [np.max(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
                        graph[:,index,1] = [np.argmax(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i]) for i in range(4)]
                    else:
                        matrix_index = [graph[j,index-1,1]*4+j for j in range(4)]
                        matrix_index = [int(k) for k in matrix_index]
                        bi = [np.max(graph[:,index-1,0]*bigram_trans_matrix[:,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
                        tri = [np.max(graph[:,index-1,0]*trigram_trans_matrix[matrix_index,i]*emi_matrix[word2id[char],i])*(2**(length>>1)) for i in range(4)]
#                        print(bi)
#                        print(tri)
#                        print('*'*60)
                        graph[:,index, 0] = np.array(tri)*alpha+np.array(bi)*(1-alpha)
                        graph[:,index, 1] = [np.argmax(graph[:,index-1,0]*trigram_trans_matrix[matrix_index,i]*emi_matrix[word2id[char],i]) for i in range(4)]

            
        endding = np.argmax(graph[:,length+1,0])
        seq = []
        seq.append(endding)
        for i in range(length,0,-1):
            seq.append(int(graph[:,:,1][:,i][seq[-1]]))
#        print(seq[1:][::-1])
        result = ''
        for index, value in enumerate(seq[1:][::-1]):
            
            if value == 0 or value == 3:
    #            print(raw_string[index])
                result += raw_string[index]+'/'
            else:
    #            print(raw_string[index])
    
                result += raw_string[index]
        real_result += result
#        print(graph)
        
    return real_result


