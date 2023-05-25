import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False, MTM=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.MTM = MTM
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
        if self.MTM:
            embedded_x = self.embeddingnet(x, c)
        else:
            embedded_x = self.embeddingnet(x)
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask
        return masked_embedding, self.mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)
    
    
class CSNet(nn.Module):
    def __init__(self, emb_model, args):
        super(CSNet, self).__init__()
        if args.model == 'MTM':
            self.embeddingnet = ConditionalSimNet(emb_model, n_conditions=args.ncondition, 
                                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein, MTM=True)
        else:
            self.embeddingnet = ConditionalSimNet(emb_model, n_conditions=args.ncondition, 
                                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)        
        
    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, c)
        embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, c)
        embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, c)
        mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm

