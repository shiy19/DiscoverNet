import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import ot
from einops import rearrange, repeat
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import permutations
import scipy.sparse as sp

# we reformalize the deep set function as with Max operator and subtract mean (optional)
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

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3 layer, 100
        :param batch_size: The experiments batch size
        """
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs, batch_size):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param x: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        c0 = Variable(torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        h0 = Variable(torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        if torch.cuda.is_available():
            c0 = c0.cuda()
            h0 = h0.cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        # residual addition
        output = output + inputs
        return output # , hn, cn


class ConceptBranch(nn.Module):
    def __init__(self, concept_type, emb_dim, out_dim, tao, normalize=True, with_label=False, pi=False):
        super(ConceptBranch, self).__init__()
        self.normalize = normalize
        self.concept_type = concept_type
        self.with_label = with_label
        self.tao = tao
        self.pi = pi
        if concept_type == 'FC':
            self.fc = nn.Sequential(nn.Linear(emb_dim * 3, 32), nn.ReLU(inplace=True), nn.Linear(32, emb_dim))
        elif concept_type == 'DS':
            self.concept_anchor = nn.Linear(emb_dim, out_dim, bias=False)
            # use deep set to select concept
            if self.with_label:
                self.set_func1 = nn.Sequential(nn.Linear(emb_dim*2+1, emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))
            else:
                self.set_func1 = nn.Sequential(nn.Linear(emb_dim*2, emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))
            self.set_func2 = nn.Sequential(nn.Linear(emb_dim, emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))
        elif concept_type == 'BiLSTM':
            self.concept_anchor = nn.Linear(emb_dim, out_dim, bias=False)
            self.bilstm_func = BidirectionalLSTM(layer_sizes=[32,32,32], vector_dim=emb_dim)
            self.linear = nn.Linear(3*emb_dim, emb_dim, bias=False)
        else:
            self.concept_anchor = nn.Linear(emb_dim, out_dim, bias=False)
            # use self-attention (transformer)
            self.slf_attn = MultiHeadAttention(1, emb_dim*2, emb_dim*2, emb_dim*2, dropout=0.5)   
            self.fc = nn.Linear(emb_dim*2, emb_dim)
            
    def forward(self, x, y, z):
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            z = F.normalize(z, p=2, dim=-1)
        if self.concept_type == 'FC':
            joint = torch.cat([x,y,z], 1)
            weight = self.fc(joint)
        elif self.concept_type == 'DS':
            if self.with_label:
                num_inst_x, emb_dim = x.shape
                joint = torch.cat([x, y, x, z, y, z], 0)
                joint = joint.view(6, num_inst_x, emb_dim)
                joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 3, emb_dim * 2)
                ones = torch.ones([num_inst_x, 1, 1], dtype=torch.float32, device=torch.device("cuda"))
                zeros = torch.zeros([num_inst_x, 1, 1], dtype=torch.float32, device=torch.device("cuda"))
                trip_label = torch.cat([zeros, ones, zeros], 1)
                joint = torch.cat([joint, trip_label], 2)   
                weight = self.set_func2((self.set_func1(joint)).max(1)[0]) # change to max here
                weight = torch.mm(weight, F.normalize(self.concept_anchor.weight, p=2, dim=1).t())
            else:
                num_inst_x, emb_dim = x.shape
                if self.pi:
                    joint = torch.cat([x, y, x, z], 0)
                    joint = joint.view(4, num_inst_x, emb_dim)
                    joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 2, emb_dim * 2)   
                else:
                    joint = torch.cat([x, y, x, z, y, z], 0)
                    joint = joint.view(6, num_inst_x, emb_dim)
                    joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 3, emb_dim * 2)   
                weight = self.set_func2((self.set_func1(joint)).max(1)[0]) # change to max here
                weight = torch.mm(weight, F.normalize(self.concept_anchor.weight, p=2, dim=1).t())
        elif self.concept_type == 'BiLSTM':
            num_inst_x, emb_dim = x.shape
            joint = torch.cat([x, y, z], 0)
            joint = joint.view(3, num_inst_x, emb_dim)
            joint = self.bilstm_func(joint, batch_size=x.shape[0])
            joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 3*emb_dim) 
            joint = self.linear(joint)
            weight = torch.mm(joint, F.normalize(self.concept_anchor.weight, p=2, dim=1).t())
        else:  # self-attention
            num_inst_x, emb_dim = x.shape
            if self.pi:
                joint = torch.cat([x, y, x, z], 0)
                joint = joint.view(4, num_inst_x, emb_dim)
                joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 2, emb_dim * 2)   
            else:
                joint = torch.cat([x, y, x, z, y, z], 0)
                joint = joint.view(6, num_inst_x, emb_dim)
                joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 3, emb_dim * 2)   
            # self-attention
            joint, _, _ = self.slf_attn(joint, joint, joint)
            joint = torch.sum(joint, 1)
            weight = self.fc(joint)
            weight = torch.mm(weight, F.normalize(self.concept_anchor.weight, p=2, dim=1).t())
        weight = F.softmax(weight/self.tao, dim=-1)
        return weight   #shape = (args.dim_embed, args.conditions)
        
        
class ConditionalSimNet(nn.Module):
    ''' output a tensor with multiple semantics'''
    def __init__(self, embeddingnet, metric_type, n_conditions, embedding_size, residual, args):
        super(ConditionalSimNet, self).__init__()
        self.embeddingnet = embeddingnet
        self.residual = residual
        self.n_conditions = n_conditions
        self.embedding_size = embedding_size
        self.metric_type = metric_type
        self.args = args
        # set different metric type
        if metric_type == 'mask':
            self.metric = torch.nn.Embedding(n_conditions, embedding_size)
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_array.fill(0.1)
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            self.metric.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
        elif metric_type == 'projection':
            # set projection
            self.metric = torch.nn.Linear(embedding_size, embedding_size * n_conditions, bias=False)
        else:
            pass           

    def forward(self, x):
        '''
        return:
            projected_x: emb per condition after projection  B*C*(n*d)
            embedded_x: the original embedding B*C*d
        '''
        embedded_x = self.embeddingnet(x)
        # apply multiple metrics over the embedding
        if self.metric_type == 'mask':
            selector = torch.LongTensor([list(range(self.n_conditions))])
            if self.args.cuda:
                selector = selector.cuda()
            metric = torch.nn.functional.relu(self.metric(selector))
            projected_x = torch.mul(embedded_x.unsqueeze(1), metric)
        elif self.metric_type == 'projection':
            num_x = x.shape[0]
            projected_x = self.metric(embedded_x).view(num_x, self.n_conditions, self.embedding_size)
        else:
            pass
        if self.residual:
            projected_x = embedded_x.unsqueeze(1) + projected_x  
        projected_x = F.normalize(projected_x, p=2, dim=-1)      
        return projected_x, embedded_x  # B x C x d

class MMetric(nn.Module):
    def __init__(self, emb_model, args):
        super(MMetric, self).__init__()
        self.num_concepts = args.ncondition
        self.embeddingnet = ConditionalSimNet(emb_model, args.metric_type, self.num_concepts, args.dim_embed, args.residual, args)
        self.concept_branch = ConceptBranch(args.concept_type, args.dim_embed, self.num_concepts, tao=args.tao, normalize=args.concept_normalize, with_label=args.with_label, pi=args.pi)
        self.args = args

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""

        # get embedding with different semantics
        embedded_x, general_x = self.embeddingnet(x)  #(B,C,D) (B,D)
        embedded_y, general_y = self.embeddingnet(y)
        embedded_z, general_z = self.embeddingnet(z)
        # get weights for concept
        weight = self.concept_branch(general_x, general_y, general_z)  #(B,C)
        
        if self.args.weight_dist:
            # apply weights over the distance
            dist_a_temp = torch.norm(embedded_x - embedded_y, p=2, dim=2)  #(B,C)
            dist_b_temp = torch.norm(embedded_x - embedded_z, p=2, dim=2)
            dist_a = torch.sum(torch.mul(dist_a_temp, weight), 1)  #element-wise
            dist_b = torch.sum(torch.mul(dist_b_temp, weight), 1)
        elif self.args.weight_loss:
            embedded_x = embedded_x.view(-1, self.args.dim_embed)  #(B*C,D)
            embedded_y = embedded_y.view(-1, self.args.dim_embed)
            embedded_z = embedded_z.view(-1, self.args.dim_embed)
            dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)   #B*C，scalar
            dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)    
        else:  
            weight = weight.unsqueeze(-1)
            # apply weights over the embedding
            embedded_x = torch.bmm(embedded_x.permute([0,2,1]), weight).squeeze(-1)
            embedded_y = torch.bmm(embedded_y.permute([0,2,1]), weight).squeeze(-1)
            embedded_z = torch.bmm(embedded_z.permute([0,2,1]), weight).squeeze(-1)  
            dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
            dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, weight


