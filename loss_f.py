import torch
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    # pad_mask = (pad_mask > 0.5)
    # p_plus = torch.stack([p,1-p],2)
    # q_plus = torch.stack([q,1-q],2)
    p_loss = F.kl_div(F.log_softmax(p,dim=-1), F.softmax(q,dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q,dim=-1), F.softmax(p,dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def multilabel_categorical_crossentropy(y_pred,y_true):
    y_pred = (1 - 2 * y_true) * y_pred    # y_true为0的项，y_pred不变，否则×-1
    y_pred_neg = y_pred - y_true * 1e12   # y_true为1的项，y_pred变成-无穷，否则不变
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # y_true为0的项，y_pred变负无穷，否则变为原来-1
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1).mean()
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1).mean()
    return neg_loss + pos_loss


def smooth_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    return (0.5 * (pred + gold).sum(1) / (corr + 1e-10)).mean()

def m_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    return ((pred + gold).sum(1) - corr).mean()

def smooth_f1_loss_linear(pred, gold):
    corr = (pred * gold).sum(1)
    return (1 - (2 * corr + 1) / ((pred + gold).sum(1) + 1)).mean()

def sample_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    pp = corr / pred.sum(1, keepdim=True)
    rr = corr / gold.sum(1, keepdim=True)
    return ((pp + rr) / (pp * rr * 2 + 1e-10) - 1).mean()

def sample_f1_loss_linear(pred, gold):
    corr = (pred * gold).sum(1)
    pp = corr / pred.sum(1, keepdim=True)
    rr = corr / gold.sum(1, keepdim=True)
    return (1 - (2 * pp * rr) / (pp + rr + 1e-10)).mean()

pu_loss_fct = lambda y_pred, y_true: - 10*(y_true*torch.log(y_pred+1e-9)).mean() - torch.log(((1-y_true)*(1-y_pred)).mean()+1e-9)