import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
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

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).cuda()
    else:
        return torch.sparse.FloatTensor(indices, values, shape)

class GraphFunc(nn.Module):
    def __init__(self, z_dim):
        super(GraphFunc, self).__init__()
        """
        DeepSets Function
        """
        self.gc1 = GraphConvolution(z_dim, z_dim * 4)
        self.gc2 = GraphConvolution(z_dim * 4, z_dim)
        self.z_dim = z_dim

    def forward(self, graph_input, graph_label=np.array([0,0,1])):
        """
        set_input, seq_length, set_size, dim
        """
        set_length, set_size, dim = graph_input.shape
        assert(dim == self.z_dim)
        # construct the adj matrix, all graphs share the same adj
        unique_class = np.unique(graph_label)
        edge_set = []
        for c in unique_class:
            current_index = np.where(graph_label == c)[0].tolist()
            if len(current_index) > 1:
                edge_set.append(np.array(list(permutations(current_index, 2))))
    
        if len(edge_set) == 0:
            adj = sp.coo_matrix((np.array([0]), (np.array([0]), np.array([0]))),
                                shape=(graph_label.shape[0], graph_label.shape[0]),
                                dtype=np.float32)
        else:
            edge_set = np.concatenate(edge_set, 0)
            adj = sp.coo_matrix((np.ones(edge_set.shape[0]), (edge_set[:, 0], edge_set[:, 1])),
                                shape=(graph_label.shape[0], graph_label.shape[0]),
                                dtype=np.float32)        
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        set_output = []
        # do GCN process
        for g_index in range(set_length):
            residual = graph_input[g_index, :]
            graph_output = F.relu(self.gc1(graph_input[g_index, :], adj))
            graph_output = F.dropout(graph_output, 0.5, training=self.training)
            graph_output = self.gc2(graph_output, adj)        
            set_output.append(residual + graph_output)
        return torch.stack(set_output)
    
class ConceptBranch(nn.Module):
    def __init__(self, concept_type, emb_dim, out_dim, normalize=True, with_label=False):
        super(ConceptBranch, self).__init__()
        self.normalize = normalize
        self.concept_type = concept_type
        self.with_label = with_label
        self.concept_anchor = nn.Linear(emb_dim, out_dim, bias=False)
        # use deep set to select concept
        self.set_func1 = nn.Sequential(nn.Linear(emb_dim*2, emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))
        self.set_func2 = nn.Sequential(nn.Linear(emb_dim, emb_dim*2), nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))
            
    def forward(self, x, y, z):
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            z = F.normalize(z, p=2, dim=-1)
        num_inst_x, emb_dim = x.shape
        joint = torch.cat([x, y, x, z, y, z], 0)
        joint = joint.view(6, num_inst_x, emb_dim)
        joint = joint.permute([1,0,2]).contiguous().view(num_inst_x, 3, emb_dim * 2)   
        weight = self.set_func2((self.set_func1(joint)).max(1)[0]) # change to max here
        # weight = torch.mm(weight, F.normalize(self.concept_anchor.weight, p=2, dim=1).t())
        # weight = F.softmax(weight, dim=-1)
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
        # set projection
        self.metric = torch.nn.Linear(embedding_size, embedding_size * n_conditions, bias=False)
        
    def forward(self, x):
        embedded_x = self.embeddingnet(x)
        # apply multiple metrics over the embedding
        num_x = x.shape[0]
        projected_x = self.metric(embedded_x).view(num_x, self.n_conditions, self.embedding_size)
        projected_x = F.normalize(projected_x, p=2, dim=-1)       
        return projected_x, embedded_x # B x C x d
    

class MMetric(nn.Module):
    def __init__(self, emb_model, args):
        super(MMetric, self).__init__()
        self.num_concepts = args.ncondition
        self.embeddingnet = ConditionalSimNet(emb_model, args.metric_type, self.num_concepts, args.dim_embed, args.residual, args)
        self.concept_branch = ConceptBranch(args.concept_type, args.dim_embed, self.num_concepts, normalize=args.concept_normalize, with_label=args.with_label)
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
        label = c
        
        # if self.args.weight_dist:
        #     # apply weights over the distance
        #     dist_a_temp = torch.norm(embedded_x - embedded_y, p=2, dim=2)  #(B,C)
        #     dist_b_temp = torch.norm(embedded_x - embedded_z, p=2, dim=2)
        #     dist_a = torch.sum(torch.mul(dist_a_temp, weight), 1)  #element-wise
        #     dist_b = torch.sum(torch.mul(dist_b_temp, weight), 1)
        # elif self.args.weight_loss:
        #     embedded_x = embedded_x.view(-1, self.args.dim_embed)  #(B*C,D)
        #     embedded_y = embedded_y.view(-1, self.args.dim_embed)
        #     embedded_z = embedded_z.view(-1, self.args.dim_embed)
        #     dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)   #B*C，scalar
        #     dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)    
        # else:  
        #     weight = weight.unsqueeze(-1)
        #     # apply weights over the embedding
        #     embedded_x = torch.bmm(embedded_x.permute([0,2,1]), weight).squeeze(-1)
        #     embedded_y = torch.bmm(embedded_y.permute([0,2,1]), weight).squeeze(-1)
        #     embedded_z = torch.bmm(embedded_z.permute([0,2,1]), weight).squeeze(-1)  
        #     dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        #     dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        # return dist_a, dist_b, weight
        return weight, label