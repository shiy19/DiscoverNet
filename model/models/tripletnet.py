import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)
            
    def forward(self, x, c):
        embedded_x = self.embeddingnet(x)
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        #masked_embedding =0* self.mask    who did it?
        masked_embedding = embedded_x * self.mask
        masked_embedding = F.normalize(masked_embedding, p=2, dim=1)
        return masked_embedding, self.mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)
    
class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class WESNet(nn.Module):
    def __init__(self, emb_model, args):
        super(WESNet, self).__init__()
        self.embeddingnet = ConditionalSimNet(emb_model, n_conditions=args.ncondition, 
                                              embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)
        self.num_concepts = args.ncondition
        self.concept_branch = ConceptBranch(self.num_concepts, args.dim_embed*3)
        self.args = args

    def forward(self, x, y, z, c):      #根据wes的原理，这个c肯定是不用的，后面也确实没用到；可以对比，class CSNet中就用到了传入的c
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""

        general_x = self.embeddingnet.embeddingnet(x)    #64*64
        general_y = self.embeddingnet.embeddingnet(y)
        general_z = self.embeddingnet.embeddingnet(z)
        # l2-normalize embeddings
        general_x = F.normalize(general_x, p=2, dim=1)
        general_y = F.normalize(general_y, p=2, dim=1)
        general_z = F.normalize(general_z, p=2, dim=1)

        feat = torch.cat((general_x, general_y), 1)
        feat = torch.cat((feat, general_z), 1)   #64*192
        weights_xy = self.concept_branch(feat)   #64*4
        embedded_x = None
        embedded_y = None
        embedded_z = None
        mask_norm = []
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx).long()
            if self.args.cuda:
                concept_idx = concept_idx.cuda()
#以下三步是给到了concept_idx计算得到的，利用CSN计算每张图片在给定条件下的embedding，也就是有监督的了，违背了WES的愿意
#并不是这样的。这里是把condition遍历了一遍，每种condition下都求一个embedding，最后加起来，并没有用到原始传入的c
            tmp_embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, concept_idx)
            mask_norm.append(masknorm_norm_x) #这个只用了x的norm也很奇怪
#基于CSN的基础上再加入WEIGHT来做attention，效果自然比csn要好。并不是这样的，因为是基于ConditionalSimNet，并不是CSNet
            weights = weights_xy[:, idx]   #64
            weights = weights.unsqueeze(1)   #64*1
            if embedded_x is None:
                embedded_x = tmp_embedded_x * weights.expand_as(tmp_embedded_x)       #64*64  *  64*64  hadamard product
                embedded_y = tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z = tmp_embedded_z * weights.expand_as(tmp_embedded_z)
            else:
                embedded_x += tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y += tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z += tmp_embedded_z * weights.expand_as(tmp_embedded_z)

        mask_norm = sum(mask_norm) / self.num_concepts
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3        #这两个norm为啥不需要像mask_norm一样把不同condition加起来？ 而是只取用最后一个condition的值？这有可能是作者写错了
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm