import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet"""
        super(ConditionalSimNet, self).__init__()
        self.embeddingnet = embeddingnet
            
    def forward(self, x, c):
        embedded_x = self.embeddingnet(x)
        return embedded_x
    
    
class SimpleNet(nn.Module):
    def __init__(self, emb_model, args):
        super(SimpleNet, self).__init__()
        self.embeddingnet = ConditionalSimNet(emb_model, n_conditions=len(args.conditions), 
                                              embedding_size=args.dim_embed)

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x = self.embeddingnet(x, c)
        embedded_y = self.embeddingnet(y, c)
        embedded_z = self.embeddingnet(z, c)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b