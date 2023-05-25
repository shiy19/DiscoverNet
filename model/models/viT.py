import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, ncondition, heads, mlp_dim, token_classification, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.con_token = nn.Parameter(torch.randn(1, ncondition, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.ncondition = ncondition
        self.token_classification = token_classification
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )


    def forward(self, img, c):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        con_tokens = repeat(self.con_token, '1 c d -> b c d', b = b)

        #choose con_token according to c
        one_hot = torch.diag(torch.ones(self.ncondition)).cuda()
        c_new_one_hot = one_hot[c]
        c_new_one_hot = c_new_one_hot.unsqueeze(1).cuda()
        con_tokens_chosen = torch.bmm(c_new_one_hot, con_tokens)

        x = torch.cat((con_tokens_chosen, cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == 'mean':
            class_emb = x[:,1:,:].mean(dim = 1) 
        else: 
            class_emb = x[:, 1]        
        class_emb = self.final_LN(class_emb)
        if self.token_classification:
            condition_emb = x[:, 0]
            condition_emb = self.final_LN(condition_emb)
            return class_emb, condition_emb
        return class_emb


class ViT_together(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, ncondition, heads, mlp_dim, token_classification, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # num_patches = (image_height // patch_height) * (image_width // patch_width) * 3
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding_inter = nn.Parameter(torch.randn(1, num_patches, dim))

        self.pos_embedding_intra_x = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_y = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_z = nn.Parameter(torch.randn(1, 1, dim))

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 4, dim))


        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 1, dim))
        self.con_token = nn.Parameter(torch.randn(1, ncondition, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.ncondition = ncondition
        self.token_classification = token_classification
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )

    def forward(self, img1, img2, img3, c):
        x = self.to_patch_embedding(img1)
        y = self.to_patch_embedding(img2)
        z = self.to_patch_embedding(img3)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = b)
        cls_tokens_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = b)
        cls_tokens_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = b)
        con_tokens = repeat(self.con_token, '1 c d -> b c d', b = b)

        #choose con_token according to c
        one_hot = torch.diag(torch.ones(self.ncondition)).cuda()
        c_new_one_hot = one_hot[c]
        c_new_one_hot = c_new_one_hot.unsqueeze(1).cuda()
        con_tokens_chosen = torch.bmm(c_new_one_hot, con_tokens)

        cls_tokens_1 = cls_tokens_1 + repeat(self.pos_embedding_intra_x, '1 1 d -> b 1 d', b=b)
        cls_tokens_2 = cls_tokens_2 + repeat(self.pos_embedding_intra_y, '1 1 d -> b 1 d', b=b)
        cls_tokens_3 = cls_tokens_3 + repeat(self.pos_embedding_intra_z, '1 1 d -> b 1 d', b=b)
        x = x + repeat(self.pos_embedding_intra_x, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        y = y + repeat(self.pos_embedding_intra_y, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        z = z + repeat(self.pos_embedding_intra_z, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 

        input = torch.cat((con_tokens_chosen, cls_tokens_1, cls_tokens_2, cls_tokens_3, x, y, z), dim=1)
        # input += self.pos_embedding[:, :(n*3 + 4)]
        input = self.dropout(input)

        input = self.transformer(input)

        if self.pool == 'mean':
            class_emb = input[:,1:,:].mean(dim = 1) 
        else: 
            class_emb_1 = input[:, 1]
            class_emb_2 = input[:, 2]  
            class_emb_3 = input[:, 3]          
        class_emb_1 = self.final_LN(class_emb_1)
        class_emb_2 = self.final_LN(class_emb_2)
        class_emb_3 = self.final_LN(class_emb_3)
        if self.token_classification:
            condition_emb = input[:, 0]
            condition_emb = self.final_LN(condition_emb)
            return class_emb_1, class_emb_2, class_emb_3, condition_emb
        return class_emb_1, class_emb_2, class_emb_3 

class ViT_global(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, ncondition, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding_inter = nn.Parameter(torch.randn(1, num_patches, dim))

        self.pos_embedding_intra_x = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_y = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_z = nn.Parameter(torch.randn(1, 1, dim))

        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.ncondition = ncondition
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )

    def forward(self, img1, img2, img3):
        x = self.to_patch_embedding(img1)
        y = self.to_patch_embedding(img2)
        z = self.to_patch_embedding(img3)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = b)
        cls_tokens_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = b)
        cls_tokens_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = b)

        cls_tokens_1 = cls_tokens_1 + repeat(self.pos_embedding_intra_x, '1 1 d -> b 1 d', b=b)
        cls_tokens_2 = cls_tokens_2 + repeat(self.pos_embedding_intra_y, '1 1 d -> b 1 d', b=b)
        cls_tokens_3 = cls_tokens_3 + repeat(self.pos_embedding_intra_z, '1 1 d -> b 1 d', b=b)
        x = x + repeat(self.pos_embedding_intra_x, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        y = y + repeat(self.pos_embedding_intra_y, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        z = z + repeat(self.pos_embedding_intra_z, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 

        input = torch.cat((cls_tokens_1, cls_tokens_2, cls_tokens_3, x, y, z), dim=1)

        input = self.dropout(input)

        input = self.transformer(input)

        if self.pool == 'mean':
            class_emb = input[:,1:,:].mean(dim = 1) 
        else: 
            class_emb_1 = input[:, 0]
            class_emb_2 = input[:, 1]  
            class_emb_3 = input[:, 2]          
        class_emb_1 = self.final_LN(class_emb_1)
        class_emb_2 = self.final_LN(class_emb_2)
        class_emb_3 = self.final_LN(class_emb_3)
        return class_emb_1, class_emb_2, class_emb_3 

class ViT_local(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, ncondition, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding_inter = nn.Parameter(torch.randn(1, num_patches, dim))

        self.pos_embedding_intra_x = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_y = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_intra_z = nn.Parameter(torch.randn(1, 1, dim))

        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.ncondition = ncondition
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )

    def forward(self, img1, img2, img3):
        x = self.to_patch_embedding(img1)
        y = self.to_patch_embedding(img2)
        z = self.to_patch_embedding(img3)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = b)
        cls_tokens_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = b)
        cls_tokens_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = b)

        cls_tokens_1 = cls_tokens_1 + repeat(self.pos_embedding_intra_x, '1 1 d -> b 1 d', b=b)
        cls_tokens_2 = cls_tokens_2 + repeat(self.pos_embedding_intra_y, '1 1 d -> b 1 d', b=b)
        cls_tokens_3 = cls_tokens_3 + repeat(self.pos_embedding_intra_z, '1 1 d -> b 1 d', b=b)
        x = x + repeat(self.pos_embedding_intra_x, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        y = y + repeat(self.pos_embedding_intra_y, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 
        z = z + repeat(self.pos_embedding_intra_z, '1 1 d -> b n d', b=b, n=n) + repeat(self.pos_embedding_inter, '1 n d -> b n d', b =b) 

        input = torch.cat((cls_tokens_1, cls_tokens_2, cls_tokens_3, x, y, z), dim=1)

        input = self.dropout(input)

        input = self.transformer(input)

        if self.pool == 'mean':
            class_emb = input[:,1:,:].mean(dim = 1) 
        else: 
            class_emb_1 = input[:, 0]
            class_emb_2 = input[:, 1]  
            class_emb_3 = input[:, 2]          
        class_emb_1 = self.final_LN(class_emb_1)
        class_emb_2 = self.final_LN(class_emb_2)
        class_emb_3 = self.final_LN(class_emb_3)
        return class_emb_1, class_emb_2, class_emb_3 

class ViT_WSCSL_weight(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, npool, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 + npool, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.con_token = nn.Parameter(torch.randn(1, npool, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.npool = npool
        self.dim = dim
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )


    def forward(self, img, weight):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        con_tokens = repeat(self.con_token, '1 c d -> b c d', b = b)

        weight = repeat(weight, 'b c -> b c d', d = self.dim)
        con_tokens_after_weight = weight*con_tokens

        x = torch.cat((cls_tokens, con_tokens_after_weight, x), dim=1)
        x += self.pos_embedding # broadcast
        x = self.dropout(x)

        x = self.transformer(x)
        class_emb = x[:, 0]        
        class_emb = self.final_LN(class_emb)
        return class_emb

class ViT_WSCSL_choose(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, nchosen, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 + nchosen, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.dim = dim
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.final_LN = nn.Sequential(
            nn.LayerNorm(dim),
        )


    def forward(self, img, con_tokens):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape   #batchsize, all_token_num

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, con_tokens, x), dim=1)
        x += self.pos_embedding # broadcast
        x = self.dropout(x)

        x = self.transformer(x)
        class_emb = x[:, 0]        
        class_emb = self.final_LN(class_emb)
        return class_emb