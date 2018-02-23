#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:06:14 2018

@author: zqun(athemeroy)
"""

import re

import numpy as np

from settings import bigram_trans_matrix, trigram_trans_matrix
from settings import word2id, emi_matrix


def viterbi(st, alpha=0.8, init=np.array([0.5, 0.5, 0, 0])):
    """
    :param st: 需要分词的字符串
    :param alpha: 仅mode=0使用：3-grams所占的比重，range(0,1)
    :param init: 初始转移概率
    :return: 分词之后的字符串
    """

    # 假设第一个字的隐状态[0.5, 0, 0, 0.5]
    def preprocess(s):
        # 为了不让概率过小导致溢出，尝试把句子通过标点符号分开
        pre_s = re.split('[; |,。，.\n]', s)
        return pre_s

    real_result = ''  # 存储最终的结果
    for string in preprocess(st):
        if not string:
            continue
        length = len(string)
        raw_string = string
        string += "Ω"  # 结尾符号

        # 把句子中不在发射矩阵（事实上是不存在索引的）的字用"≈"替换（也就是UNK）
        for i in string:
            if not word2id.get(i):
                string = string.replace(i, "≈")

        # graph是一个双重矩阵，一重表示到该节点的最大概率
        # 二重表示到底是哪个上层节点过来的
        graph = np.zeros((4, length + 2, 2))

        # 给起始点一个初始概率，默认是S:0.5, B:0.5, M:0, E:0
        # 因为第一个字不是单字词就是一个词的起始，不可能是词中间或者是结尾
        graph[:, 0, 0] = init

        for index, char in enumerate(string):
            # 处理第一个字
            if index == 0:
                graph[:, 0, 0] = graph[:, 0, 0] * emi_matrix[word2id[char]]
            else:
                if index == 1:
                    # 处理第二个字，这是不能使用二阶马尔科夫模型，使用一阶马尔科夫模型代替
                    graph[:, index, 0] = [np.max(
                        graph[:, index - 1, 0] * bigram_trans_matrix[:, i] * emi_matrix[word2id[char], i]) * (
                                              2 ** (length >> 1)) for i in range(4)]
                    graph[:, index, 1] = [
                        np.argmax(graph[:, index - 1, 0] * bigram_trans_matrix[:, i] * emi_matrix[word2id[char], i])
                        for i in range(4)]
                else:
                    # 处理后面的字：
                    # 使用二阶马尔科夫模型估计概率
                    # 首先准备三阶的发射概率：通过前一层第二重得到前一层某节点在前前层的来源，乘4得到发射矩阵的大致行数
                    # 接下来加上自身，得到发射矩阵确定的行数
                    matrix_index = [graph[j, index - 1, 1] * 4 + j for j in range(4)]
                    matrix_index = [int(k) for k in matrix_index]
                    bi = np.array([bigram_trans_matrix[:, i] for i in range(4)]) * (1 - alpha)
                    tri = np.array([trigram_trans_matrix[matrix_index, i] for i in range(4)]) * alpha
                    # bi = [np.max(graph[:, index - 1, 0] * bigram_trans_matrix[:, i] * emi_matrix[word2id[char], i]) * (
                    # 2 ** (length >> 1)) for i in range(4)]
                    # tri = [np.max(graph[:, index - 1, 0] * trigram_trans_matrix[matrix_index, i] * emi_matrix[
                    #     word2id[char], i]) * (2 ** (length >> 1)) for i in range(4)]
                    graph[:, index, 0] = [np.max(
                        graph[:, index - 1, 0] * (bi[i] + tri[i]) * emi_matrix[word2id[char], i])
                        for i in range(4)]
                    graph[:, index, 1] = [np.argmax(
                        graph[:, index - 1, 0] * (bi[i] + tri[i]) * emi_matrix[word2id[char], i])
                        for i in range(4)]
        # print(graph)

        endding = np.argmax(graph[:, length + 1, 0])
        seq = list()
        seq.append(endding)
        for i in range(length, 0, -1):
            seq.append(int(graph[:, :, 1][:, i][seq[-1]]))
        # print(seq[1:][::-1])
        result = ''
        for index, value in enumerate(seq[1:][::-1]):

            if value == 0 or value == 3:
                #            print(raw_string[index])
                result += raw_string[index] + '/'
            else:
                #            print(raw_string[index])
                result += raw_string[index]
        real_result += result + '\n'

    return real_result


if __name__ == '__main__':
    print(viterbi('''
    明确规定监察委员会组成人员依法产生后，应当进行宪法宣誓、宣誓仪式应当奏唱中华人民共和国国歌。
    报道称，密歇根州卫生和公共服务部没有透露这名工作人员的姓名，但指出从去年5月到今年1月底之间他可能接触了多达600人。
    不过，由于深铁、万科管理层及盟友的持股合计超过40%，而该议案只需获得参与投票的半数赞成票即可通过，预计中小股东将很难阻止刘姝威的加薪。
    钴是目前全球范围内采用最广泛的锂电池的重要原料之一，主要用于三元锂电池的正极材料，钴可以明显提升锂电池的能量密集度，目前全世界钴的产量约有一半以上用于制造锂钴电池，四分之一被用于智能手机等电子设备。
    她的半尾眼线一般都从瞳孔靠近眼头外侧相对的正上方开始，才去从极细到微粗的渐变方式，最后在眼尾做上扬处理，修饰了她微微下垂的眼角，俏皮的刚好。
    美国哈佛医学院心脏科教授克多·格尔威治博士指出，洋葱含有大量保护心脏的类黄酮，每天生吃半个可增加心脏病人约30%的“好胆固醇”。尤其在吃烤肉这样不怎么健康的食物时，里面的洋葱就像你的“救命草”。
    当移动支付市场形成双雄争霸，阿里、腾讯双方激战正酣之时，银联发布的银行业统一App“云闪付”入场，以春节红包的形式发起攻坚战。“云闪付”堪称含着金汤匙出生的支付工具，银联的用户基数也远超微信和支付宝。2018年，移动支付线下混战将比以往更甚。
    “5个月时间，关于币和链的自媒体，就出现了数千家。”自媒体数据监测平台的负责人称，其中每日更新的，也有上千家，“就算是自媒体黄金时代，也没看到如此火热的崛起浪潮”。
''', alpha=0.8))
