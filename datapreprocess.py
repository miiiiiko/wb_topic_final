import pandas as pd
from utils.ljqpy import LoadJsons,SaveJsons
import random
import unicodedata
import zhconv,emoji

dpath  = './dataset/raw_data/train.csv'
df = pd.read_csv(dpath,  sep='\t', encoding="utf-8")

def transfer_to_json(df,out_path):
    '''
    将csv文件转化为json文件，方便后续调用
    '''
    data = []
    for i in range(len(df)):
        l = {}
        l["id"] = int(df.iloc[i,0])
        l["text"] = df.iloc[i,1]
        l["label"] = df.iloc[i,2].split("，")
    #     l = json.dumps(l, ensure_ascii=False)
    #     fw.write(l + '\n')
    # fw.close()
        data.append(l)
    SaveJsons(data,out_path)
    return



def sep_data(file_path:str):  
    '''
    将数据随机打乱并切分成训练集和验证集
    '''
    data = []
    for xx in LoadJsons(file_path):  # 数据格式.json
        xx['text_normd'] = xx['text'].replace('\u200b','')
        xx['text_normd'] = unicodedata.normalize('NFKC', xx['text_normd'])  # 同时清洗数据，并保存到新的字段中
        data.append(xx)
        
    random.shuffle(data)
    train = data[5000:]; val = data[:5000]
    SaveJsons(train,'./dataset/train_normd.json')
    SaveJsons(val,'./dataset/val_normd.json')


cc = {'𝓪':'a','𝒶':'a','𝒜':'A','𝓐':'A','𝒂':'a','ⓐ':'a','𝐴':'A','𝑎':'a','𝗮':'a','𝗔':'A','𝟬':'0'}
fconv = {}
for x, y in cc.items():
    mx = 10 if y == '0' else 26
    for i in range(mx): 
        fconv[chr(ord(x)+i)] = chr(ord(y)+i)

def ConvertFlower(zz):
    '''
    转换花体
    '''
    newz = []
    for z in zz: newz.append(fconv.get(z, z))
    return ''.join(newz)

def Normalize(z):
    '''
    将花体转换成正常英文、数字，删除表情，繁体转简体
    '''
    z = ConvertFlower(z)
    return zhconv.convert(emoji.replace_emoji(z, replace=''),'zh-cn')


if __name__ == '__main__':
    random.seed(1305)
    transfer_to_json(df,'./dataset/train.json')
    sep_data('./dataset/train.json')