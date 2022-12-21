import sys
sys.path.append("./")
import torch
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
from utils import sortlabel,ljqpy, pt_utils
from utils.sortlabel import llist
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from model import Model
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
import time
from datapreprocess import Normalize
from loss_f import compute_kl_loss,multilabel_categorical_crossentropy,smooth_f1_loss_linear,pu_loss_fct
# import emoji
# import zhconv
# from ljqpy import LoadJsons

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_name = 'hfl/chinese-roberta-wwm-ext'
# model_name_l = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
# source = ljqpy.LoadJsons('./dataset/train.json')
# llist = sortlabel.TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)


def load_data(fn):
    '''
    加载数据
    '''
    return [(x["text_normd"], x["label"]) for x in ljqpy.LoadJsons(fn)]

datadir = './dataset'

# 加载训练集与验证集
xys = [load_data(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]


def label2vec(targrtlabels:list,dims= llist.get_num()):
    '''
    将标签列表转化成one-hot形式向量
    '''
    lab_vec = torch.zeros(dims)
    for label in targrtlabels:
        loc = llist.get_id(label)
        if loc != -1:
            lab_vec[loc] = 1
    return lab_vec


class MyDataset(Dataset):
    def __init__(self,data, requires_index = False):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            text = d[0]
            # 由于后期才想起来可以再次处理文本的表情等，故前期训练期间没有使用Normalize,上交的模型参数有的是用Normalize的文本训练的，会标注noemo，
            # 否则就是没做normalize
            text = Normalize(d[0])
            label = label2vec(d[1],dims= llist.get_num())
            # requires_index是为以后挑出错误分类的样本做准备
            if requires_index:
                self.data.append([text,label,torch.tensor([i])])
            else:
                self.data.append([text,label])

    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    if len(batch[0]) ==2:
        return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.stack([x[1] for x in batch], 0))
    else:
        return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.stack([x[1] for x in batch], 0),
            torch.cat([x[2] for x in batch]))
        
# (b, max_len)

def plot_learning_curve(record,pic_n):
    '''
    训练作图所用函数
    '''
    y1 = record['train_loss']
    y2 = record['val_f1']
    x1 = np.arange(1,len(y1)+1)
    x2 = x1[::int(len(y1)/len(y2))]
    fig = figure(figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1, c = 'tab:red', label = 'train_loss')
    ax2 = ax1.twinx()
    ax2.plot(x2,y2, c='tab:cyan', label='val_f1')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('val_f1')
    plt.title('Learning curve')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig(pic_n)
    return

def cal_hour(seconds):
    '''
    将秒数转换成小时，分钟，秒数的形式，方便记录训练时间
    '''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)



def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  
    '''
    模型的主训练函数
    '''
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        print(f'\nEpoch {epoch+1} / {epochs}:')
        if accelerator:
            pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
        else: 
            pbar = tqdm(train_dl, total=epoch_len)
        metricsums = {}
        iters, accloss = 0, 0
        for ditem in pbar:
            metrics = {}
            loss = train_func(model, ditem)
            if type(loss) is type({}):
                metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
                loss = loss['loss']
            iters += 1; accloss += loss
            optimizer.zero_grad()
            if accelerator: 
                accelerator.backward(loss)
            else: 
                loss.backward()
            optimizer.step()
            if scheduler:
                if accelerator is None or not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
            for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
            infos = {'loss': f'{accloss/iters:.4f}'}
            for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            pbar.set_postfix(infos)
            if epoch_len and iters > epoch_len: break
        pbar.close()
        if test_func:
            if accelerator is None or accelerator.is_local_main_process: 
                model.eval()
                accu,prec,reca,f1 = test_func()
                if f1 >=best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f})")
                    best_f1 = f1



if __name__ == '__main__':
    record = {"train_loss":[],"val_f1":[]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 采用不同的损失函数与模型训练了多个模型参数
    mfile0 = './output/ljq/bert_wb.pt' # 此处可修改为任何训练了一段时间的模型参数
    # 是否要在原来的基础上继续训练，如是，则加载mfile0的参数,否则删除mflie=xxx项即可
    model = Model(model_name, llist.get_num()).to(device)
    # model = Model(model_name, llist.get_num(),activation=True).to(device)
    # 为了避免过拟合对ransformer层进行锁层，即冷冻部分参数，此处是否锁层、锁哪些层训练出不同的模型参数，可以加快模型的训练速度
    pt_utils.lock_transformer_layers(model.encoder, 10)
    ds_train, ds_test = MyDataset(xys[0]), MyDataset(xys[1])
    print('loading data completed')
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=collate_fn)
    print("dataloader completed")
  
    print("finish loading model")
    # 最后保存参数的地址,为了防止路径错误，请在当前文件夹下建立output文件夹
    mfile = './output/128_base7.ckpt'

    alpha = 4
    epochs = 40
    total_steps = len(dl_train) * epochs

    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 5e-5, total_steps)
    loss_func = multilabel_categorical_crossentropy
    start_time = time.time()
    val_time = 0
    def train_func(model, ditem):
        xx, yy = ditem[0].to(device), ditem[-1].to(device)
        zz1 = model(xx)
        zz2 = model(xx)

        # 此处采用了r-drop的训练技巧
        multi_loss = 0.5*(loss_func(zz1.float(), yy.float()) + loss_func(zz2.float(), yy.float()))
        kl_loss = compute_kl_loss(zz1, zz2)
        loss = multi_loss + alpha * kl_loss

        record["train_loss"].append(loss.item())
        return {'loss': loss}

    def test_func(): 
        global val_time
        t1 = time.time()
        yt, yp = [], []
        model.eval()
        with torch.no_grad():
            for xx,_,_, yy in dl_test:
                xx, yy = xx.to(device), yy
                zz = (model(xx).detach().cpu() > 0).float().cpu()
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            yt = torch.stack(yt,0).numpy().astype('int64')
            yp = torch.stack(yp,0).numpy().astype('int64')
            accu = metrics.accuracy_score(yt,yp)
            prec = metrics.precision_score(yt,yp,average='samples',zero_division=0)
            reca = metrics.recall_score(yt,yp,average='samples',zero_division=0)
            f1 = metrics.f1_score(yt,yp,average='samples',zero_division=0)
            
            record["val_f1"].append(f1)
            # f1_1d = metrics.f1_score(yt.unsqueeze(1).numpy().astype('int64'),yp.unsqueeze(1).numpy().astype('int64'),average='samples')
        print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
        model.train()
        t2 = time.time()
        val_time += t2-t1
        return accu,prec,reca,f1

    print('Start training!')
    train_model(model, optimizer, dl_train, epochs, train_func, test_func, scheduler=scheduler, save_file=mfile)
    end_time = time.time()
    val_time = val_time/epochs
    total_time = end_time-start_time
    total_time = total_time/epochs
    train_time = total_time - val_time
    total_time,train_time,val_time = cal_hour(total_time),cal_hour(train_time),cal_hour(val_time)
    print(f'Train_time:{train_time}, Val_time:{val_time}, total_time:{total_time}')
    # 文件夹下必须有output文件夹，否则没法保存参数与保存图片
    plot_learning_curve(record,'./output/128base7_rdrop')
    print('done')
