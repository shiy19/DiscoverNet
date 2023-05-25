import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.embeddingnet = embeddingnet
       
    def forward(self, x, c):
        embedded_x = self.embeddingnet(x)
        embedded_x = F.normalize(embedded_x, dim=-1, p=2)
        return embedded_x
    
    
class DummyNet(nn.Module):
    def __init__(self, emb_model, args):
        super(DummyNet, self).__init__()
        self.embeddingnet = ConditionalSimNet(emb_model)

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