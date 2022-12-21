import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel

class MLP(nn.Module):
    def __init__(self, n_in, n_out, activation=False): 
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        if activation:  # 是否加入sigmoid层
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class Model(nn.Module):  # 模型基础结构，Bert+线性全连接层
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext',n=1400,encoder_type = 'cls',activation=False,mfile0 = None):
        super(Model, self).__init__()
        self.encoder_type = encoder_type
        self.encoder = BertModel.from_pretrained(model_name)
        if mfile0:
            self.encoder.load_state_dict(torch.load(mfile0), strict=False)
        self.pred = MLP(768,n,activation=activation)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def forward(self,inputs):
        x = self.encoder(inputs)
        if self.encoder_type == "cls":
            x = x.last_hidden_state[:,0]
        if self.encoder_type == "pooler":
            x = x.pooler_output
        x = self.pred(x)
        return x
