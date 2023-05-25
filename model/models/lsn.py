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
                self.masks = torch.nn.Embedding(n_conditions, embedding_size).cuda()
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
        masked_embedding = embedded_x * self.mask
        masked_embedding = F.normalize(masked_embedding, p=2, dim=1)
        return masked_embedding, self.mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)

class LSNet(nn.Module):
    def __init__(self, emb_model, args):
        super(LSNet, self).__init__()
        self.embeddingnet = ConditionalSimNet(emb_model, n_conditions=args.ncondition, 
                                              embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)
        self.num_concepts = args.ncondition
        self.args = args
        self.criterion = torch.nn.MarginRankingLoss(margin = args.margin)      

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        mask_norm, embed_norm, mask_embed_norm = [], [], []
        dist_a, dist_b = [], []
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx).long()
            if self.args.cuda:
                concept_idx = concept_idx.cuda()

            tmp_embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, concept_idx)
            mask_norm.append(masknorm_norm_x)
            embed_norm.append((embed_norm_x + embed_norm_y + embed_norm_z) / 3)
            mask_embed_norm.append((tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3)
            dist_a.append(F.pairwise_distance(tmp_embedded_x, tmp_embedded_y, 2))
            dist_b.append(F.pairwise_distance(tmp_embedded_x, tmp_embedded_z, 2))
            
        # compute loss for each concept
        loss_set = []
        target = torch.ones_like(dist_a[0])
        if self.args.cuda:
            target = target.cuda()          
        for idx in range(self.num_concepts):
            loss_set.append(self.criterion(dist_a[idx], dist_b[idx], target))
        loss_set = torch.stack(loss_set)
        #print(loss_set)    
        output_loss = torch.min(loss_set)
        #print(output_loss)
        select_concept = torch.argmin(loss_set)
        dist_a = dist_a[select_concept]
        dist_b = dist_b[select_concept]
        mask_norm = mask_norm[select_concept]
        embed_norm = embed_norm[select_concept]
        mask_embed_norm = mask_embed_norm[select_concept]
        return output_loss, dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm