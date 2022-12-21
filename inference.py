import sys
sys.path.append('./')
import pandas as pd
import unicodedata
from utils.ljqpy import SaveJsons,LoadJsons
from transformers import BertTokenizer
from torch.utils.data import DataLoader,Dataset
import torch
from tqdm import tqdm
from model import Model
from utils import sortlabel,pred_helper
from datapreprocess import Normalize
from copy import deepcopy
from utils.pred_helper import Val0_Classify
from utils.sortlabel import llist
# from base1.merge_func import Val0_Classify
# 采用目前最好的指标，多分类+argmax
model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
# 测试集要求id和label
def transfer_test(dpath,outpath):
    '''
    将测试集转换成json文件
    '''
    data = []
    df = pd.read_csv(dpath,  sep='\t', encoding="utf-8")
    for i in range(len(df)):
        l = {}
        l['text_normd'] = df.iloc[i,1].replace('\u200b','')
        l['text_normd'] =  unicodedata.normalize('NFKC', l['text_normd'])
        l['ID'] = int(df.iloc[i,0])
        data.append(l)
    SaveJsons(data,outpath)
    return 

def load_test(fn):
    '''
    有的模型是在原始文本上训练的，有的文本是在Normalize后的文本上训练的，故需加载两种文本
    '''
    return [(x["text_normd"],Normalize(x["text_normd"]), x["ID"]) for x in LoadJsons(fn)]

class TestDS(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            text1 = d[0]
            text2 = d[1]
            id = d[2]
            id = torch.tensor([id])
            i = torch.tensor([i])
            self.data.append([text1,text2,id,i])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_test(batch):
    z1 = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    z2 = tokenizer([d[1] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    return (z1.input_ids,
            z2.input_ids,
            torch.cat([x[2] for x in batch], 0),
            torch.cat([x[3] for x in batch], 0))

class Inference:
    def __init__(self, model, mfile_l, llist, device=torch.device('cuda'),n=11):
        self.device = device
        self.model_l = [deepcopy(model).to(device) for i in range(len(mfile_l))]
        for i,mfile in enumerate(mfile_l):
            self.model_l[i].load_state_dict(state_dict = torch.load(mfile, map_location=device))
            self.model_l[i].eval()
        self.tl = llist
        self.n = n
        
    def infer_on_test(self,test_data,test_dl):
        # res: id to label list
        res = {}
        # res2 = []
        Classifier = Val0_Classify(llist)
        with torch.no_grad():
            for xx1,xx2,ids,indices in tqdm(test_dl):
                # print(indices)
                xx1,xx2 = xx1.to(self.device),xx2.to(self.device)
                scores_l = torch.zeros(xx1.shape[0],self.tl.get_num(),len(self.model_l))
                # print(scores_l.shape)
                for i,model in enumerate(self.model_l):
                    # model_l里，前n个是用原始文本训练的模型，后面的是用Normalized过的文本训练的模型
                    if i < self.n:
                        scores_l[...,i] = model(xx1).detach().cpu()
                    else:
                        scores_l[...,i] = model(xx2).detach().cpu()
                # 对n个模型的成绩求平均
                scores = scores_l.mean(-1)
                # 对于总是共同出现的标签对，其分数取两者较大值
                scores = pred_helper.transfer(scores)
                # threshold设为0.5
                zz = (scores > 0.5).float().cpu()
    
                for idx,z in enumerate(zz):
                    # 取出文本单独判断，是否可以通过硬匹配给文本贴上话题标签
                    text = test_data[indices[idx]][0]
                    ind_plus = pred_helper.check_text(text)
                    for j in ind_plus:
                        z[j] = 1
                    z_vec = z
                    # 如果输出全0，那么使用特殊的规则 
                    if sum(z) == 0:
                        z_vec = Classifier.fun(scores[idx])
                    # 取出标签index    
                    iis = z_vec.nonzero().squeeze(1)
                    res[int(ids[idx].item())] = [self.tl.get_token(i) for i in iis]
                    # res2.append({'text':text,'pred':[(self.tl.get_token(i),scores[idx][i].item()) for i in iis]})
        # SaveJsons(res2, './output/result/infer_score_1.json')
        return res


def write_to_csv(id2l, out_path):
    '''
    将预测结果写入submission.csv
    '''
    data = {'ID':[], 'Label':[]}
    for k,v in id2l.items():
        data['ID'].append(str(k))
        data['Label'].append('，'.join(v))
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=0,sep='\t')
    return



if __name__ == '__main__':
    transfer_test('./dataset/test.csv','./dataset/test.json')
    # model_name = 'hfl/chinese-roberta-wwm-ext'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    test_data = load_test('./dataset/test.json')
    test_ds = TestDS(test_data)
    test_dl = DataLoader(test_ds,collate_fn=collate_test,batch_size=128)
    model = Model(activation=True)
    # 请创建model_states文件夹, 将模型参数解压到该文件夹下
    file_l =  ['wb_base_retrain_lock7_diceloss_normd2_8981.pt','wb_base2_noseg_normd2.pt'] + ['base5.ckpt', '256base1.pt','128_base7.ckpt', '128_base7_plus5.ckpt','wb_base_noseg_normd1_8855.pt',
    'wb_base_retrain_lock7_diceloss2_normd1_8846.pt','wb_base_retrain2_lock10_diceloss_normd1_8871.pt','wb_base_retrain2_lock7_diceloss_normd1_8858.pt',
    'wb_base_retrain_lock8_normd1_8844.pt','wb_base_retrain_lock7_diceloss_normd1_8867.pt','wb_base_retrain_lock6_normd1_8859.pt',
    'base7noemo.ckpt','base7noemo_n2.ckpt']
    mfile_l = ['./model_states/' + s for s in file_l]

    # mfile_l =  ['./output/ljq/wb_base_retrain_lock7_diceloss_normd2_8981.pt','./output/ljq/wb_base2_noseg_normd2.pt'] + ['./output/base5/base5.ckpt', './output/base1/256base1.pt','./output/base7/128_base7.ckpt', './output/extra/128_base7_plus5.ckpt','./output/base7/wb_base_noseg_normd1_8855.pt',
    # './output/ljq/wb_base_retrain_lock7_diceloss2_normd1_8846.pt','./output/ljq/wb_base_retrain2_lock10_diceloss_normd1_8871.pt','./output/ljq/wb_base_retrain2_lock7_diceloss_normd1_8858.pt',
    # './output/ljq/wb_base_retrain_lock8_normd1_8844.pt','./output/ljq/wb_base_retrain_lock7_diceloss_normd1_8867.pt','./output/ljq/wb_base_retrain_lock6_normd1_8859.pt',
    # './output/base7/base7noemo.ckpt','./output/base7/base7noemo_n2.ckpt']
    # print(len(mfile_l))
    source = LoadJsons('./dataset/train.json')
    # llist = sortlabel.TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)
    Infer = Inference(model,mfile_l,llist,torch.device('cuda'),n=13)
    res = Infer.infer_on_test(test_data,test_dl)
    write_to_csv(res,'submission.csv')
    # print(list(res.values())[:5])
    # print(list(res.keys())[0])
