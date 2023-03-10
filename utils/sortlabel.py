import sys
sys.path.append('./')
import os
import numpy
from collections import defaultdict
from utils.ljqpy import LoadJsons,TokenList,FreqDict2List,SaveCSV,LoadCSV
# import torch

class TokenList:  # 将所有的标签按频数从高到低排序，保存到文件中
    def __init__(self, file, low_freq=1, source=None, func=None, save_low_freq=1, special_marks=[]):
        if not os.path.exists(file):
            tdict = defaultdict(int)
            for i, xx in enumerate(special_marks): tdict[xx] = 100000000 - i
            for xx in source:
                for token in func(xx): tdict[token] += 1
            tokens = FreqDict2List(tdict)  # already decline sorted
            tokens = [x for x in tokens if x[1] >= save_low_freq]
            SaveCSV(tokens, file)
        self.id2t = [ '<UNK>'] + [x for x,y in LoadCSV(file) if float(y) >= low_freq]
        self.t2id = {v:k for k,v in enumerate(self.id2t)}
        self.id2freq = [int(y) for x,y in LoadCSV(file) if float(y) >= low_freq]
        self.freq2id = {v:k+1 for k,v in enumerate(self.id2freq)}
    def get_id(self, token): return self.t2id.get(token, -1)
    def get_token(self, ii): return self.id2t[ii]
    def get_num(self): return len(self.id2t)

def label2vec(targrtlabels:list,llist:TokenList):  # 读取每个文本的标签，并生成长度为标签个数的向量，其中该文本对应标签的值为1，其余为0
    lab_vec = numpy.zeros(llist.get_num())
    for label in targrtlabels:
        loc = llist.get_id(label)
        lab_vec[loc] = 1.0
    return lab_vec

def label2id(targrtlabels:list,llist:TokenList):  # 读取每个文本的标签，并获得它们在上面文件中的排序
    locs = []
    for label in targrtlabels:
        locs.append(llist.get_id(label))
    return locs
def loc(num:int, llist:TokenList):
    '''
    返回llist中第一个频数为num的id，llist[:x]得到的是频数大于num的列表，[x:]则为小于等于
    '''
    for inx,i in enumerate(llist.id2freq): 
        if i < num+1: return inx+1
        elif inx == len(llist.id2freq)-1: return inx+1
def sep_point(freq_list:list):  # 返回一系列频数的位置id
    return [loc(item) for item in freq_list]

source = LoadJsons('./dataset/train.json')
llist = TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)
