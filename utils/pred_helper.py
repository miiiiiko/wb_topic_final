import sys
sys.path.append('./')
import torch
from utils import ljqpy
from utils.sortlabel import llist


all_data = ljqpy.LoadJsons('./dataset/train.json')
label_matrix = torch.zeros(1400,1400)

# 计算共现矩阵，将总是一起出现的标签对绑定在一起
for d in all_data:
    label_list = d['label']
    for i in range(len(label_list)):
        for j in range(len(label_list)):
            l1 = llist.get_id(label_list[i])
            l2 = llist.get_id(label_list[j])
            label_matrix[l1][l2] += 1

label_matrix = torch.stack([ele/ele.sum() for ele in label_matrix])
label_pair = (label_matrix == 1).nonzero()
# print(len(label_pair))
# print(label_pair)
def transfer(scores):
    '''
    scores是batch_size*n_labels的tensor，上已经计算总是一起出现的标签对label_pair，
    对于每一对标签，在推理时该标签的分数为该标签对分数较大的那个
    '''
    for i in range(len(label_pair)):
        i,j = label_pair[i][0],label_pair[i][1]
        res = torch.max(scores[...,i],scores[...,j])
        scores[...,i],scores[...,j] = res,res
    return scores
# scores = torch.tensor([[0]*1400,[1]*1400])

# transfer(scores)
def transfer_z(z):
    '''
    如果一个标签为1，标签对都应被判断为1
    '''
    for i,j in label_pair:
        if z[i] == 1 or z[j] == 1:
            z[i],z[j] = 1,1
    return z

# 人工添加了一些规则，对于模型仍然预测不好的标签，找出其关键词，进行硬匹配
lab2word = {'label_895459':'入境','label_582805':'情侣头像','label_493566':'萧邦',
            'label_766890':'单身久了', 'label_819030':'张俪','label_1219604':'堂食','label_1340977':'民族匠心','label_361083':'反诈老陈',
            'label_658167':'迷你世界','label_137439':'恋恋北极星','label_1390048':'航空公司','label_207137':'李紫婷','label_161121':'金译团',
            'label_1003895':'光子','label_161726': '爱情电影频道','label_102402': '六一儿童节',"label_1064693":"韩国炸鸡",'label_312604':'华为深圳','label_596430':'榴莲',
            'label_1516519':'姜栋元','label_247451':'天猫618','label_1313425':'天猫618','label_206611':'小麦','label_1429838':'夏娃','label_714824':'金湾'}


def check_text(text,lab2word=lab2word):
    '''
    检查文本，若文本含有关键词，即判断文本有该关键词对应的标签
    '''
    indices = []
    for k,v in lab2word.items():
        if v in text:
            indices.append(llist.get_id(k))
    return indices


class Val0_Classify():  # 对预测结果判定规则后，使用该方法进行重新预测
    def __init__(self,llist):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tl = llist
        self.fun = self.argmax_func

       
    def argmax_func(self,score_i):
        pred_vec = torch.zeros_like(score_i)
        threshold = score_i.max()*0.4
        for idx in (score_i > threshold).float().nonzero().squeeze(1):
            pred_vec[idx] = 1
        return pred_vec
