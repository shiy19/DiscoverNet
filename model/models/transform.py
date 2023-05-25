import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Also adapt the global classifier with Generalized Dict
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        # combine local and global key and value
        k = torch.cat([q, k], 1)
        v = torch.cat([q, v], 1)
        len_k = len_k + len_q
        len_v = len_v + len_q        
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn, log_attn
    
class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Transform(nn.Module):
    def __init__(self, emb_model, args):
        super(Transform, self).__init__()
        self.encoder = emb_model
        # self.num_concepts = len(args.conditions)
        self.args = args
        self.slf_attn = MultiHeadAttention(1, args.dim_embed, args.dim_embed, args.dim_embed, dropout=0.2)     
        self.metric_base = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(1, 128, args.dim_embed)))
    
    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        
        num_inst_x, num_inst_y, num_inst_z = x.shape[0], y.shape[0], z.shape[0]
        assert(num_inst_x == num_inst_y and num_inst_x == num_inst_z)
        whole_emb = self.encoder(torch.cat([x, y, z], 0))
        whole_emb = whole_emb.view(3, num_inst_x, self.args.dim_embed)
        whole_emb = whole_emb.permute([1,0,2])
        # x_emb, y_emb, z_emb = whole_emb.split(num_inst_x, dim=0)
        
        # do transformation on (x,y) and (x,z)
        adapt_emb, _, _ = self.slf_attn(whole_emb, self.metric_base.repeat(num_inst_x, 1, 1), self.metric_base.repeat(num_inst_x, 1, 1)) # num_task * num_way * dim
        
        # l2-normalize embeddings
        adapt_emb = F.normalize(adapt_emb, p=2, dim=2)
        embedded_x, embedded_y, embedded_z = adapt_emb[:,0,:], adapt_emb[:,1,:], adapt_emb[:,2,:]
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, None, torch.sum(torch.norm(adapt_emb, p=2, dim=2)), None