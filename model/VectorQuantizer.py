# import torch
# from torch import nn
# from torch.nn import functional as F

# class VectorQuantizer(nn.Module):
#     """
#     VQ-VAE layer: Input any tensor to be quantized. 
#     Args:
#         embedding_dim (int): the dimensionality of the tensors in the
#           quantized space. Inputs to the modules must be in this format as well.
#         num_embeddings (int): the number of vectors in the quantized space.
#         commitment_cost (float): scalar which controls the weighting of the loss terms (see
#           equation 4 in the paper - this variable is Beta).
#     """
#     def __init__(self, embedding_dim, num_embeddings, beta):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = beta
#         self.init = False
        
#         # initialize embeddings
#         self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
       
#     def forward(self, x):
#         # [B, C, H, W] -> [B, H, W, C]
#         # print("x.shape ", x.shape)
        
#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # TODO 这里训练时要注释掉
#         # device = torch.device('cuda:0')
#         # print("x", x)
#         # [B, H, W, C] -> [BHW, C]
#         # x = x[:, :96, :]
#         # if not self.init:
#         #     self.init_emb(x)
#         flat_x = x.reshape(-1, self.embedding_dim)
#         # flat_x = flat_x.to(device)
        
#         encoding_indices = self.get_code_indices(flat_x)
#         quantized = self.quantize(encoding_indices)
#         quantized = quantized.view_as(x) # [B, H, W, C]

#         # min_distance, x_l = self.get_code_indices(flat_x)
#         # quantized = self.quantize(x_l)
#         # quantized = quantized.view_as(x) # [B, H, W, C]
        
#         # embedding loss: move the embeddings towards the encoder's output
#         q_latent_loss = F.mse_loss(quantized, x.detach())
#         # commitment loss
#         e_latent_loss = F.mse_loss(x, quantized.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
#         # loss = torch.norm(quantized.detach() - x) ** 2 / np.prod(x.shape) * self.commitment_cost

#         # Straight Through Estimator
#         quantized = x + (quantized - x).detach()
        
#         # quantized = quantized.permute(0, 3, 1, 2).contiguous()
#         # quantized = quantized.permute(0, 2, 1).contiguous()
#         # quantized = torch.squeeze(quantized)
#         return quantized, loss
    
#     def get_code_indices(self, flat_x):
#         # compute L2 distance
#         # print("self.embeddings.weight ", self.embeddings.weight)
#         distances = (
#             torch.sum(flat_x ** 2, dim=1, keepdim=True) +
#             torch.sum(self.embeddings.weight ** 2, dim=1) -
#             2. * torch.matmul(flat_x, self.embeddings.weight.t())
#         ) # [N, M]
#         encoding_indices = torch.argmin(distances, dim=1) # [N,]
#         return encoding_indices

#         # distances = (
#         #     torch.sum(flat_x ** 2, dim=-1, keepdim=True) +
#         #     torch.sum(self.embeddings.t().to('cuda') ** 2, dim=0, keepdim=True) -
#         #     2. * torch.matmul(flat_x, self.embeddings.t().to('cuda'))
#         # ) # [N, M]

#         # min_distance, x_l = torch.min(distances, dim=-1) # (min, min_indices)
#         # # encoding_indices = torch.argmin(distances, dim=1) # [N,]
#         # return min_distance, x_l
    
#     def quantize(self, encoding_indices):
#         """Returns embedding tensor for a batch of indices."""
#         return self.embeddings(encoding_indices) 
#         # x = F.embedding(encoding_indices, self.embeddings)
#         # return x

import torch as t
from torch import nn
from torch.nn import functional as F
import numpy as np

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
class VectorQuantizer(nn.Module):
    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        # TODO
        # self.register_buffer('k', t.zeros(self.k_bins, self.emb_width).cuda())
        self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]
        # dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = t.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = t.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    # x_l 向量的坐标
    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with t.no_grad():
            # Calculate new centres
            x_l_onehot = t.zeros(
                k_bins, x.shape[0], device=x.device)  # k_bins, N * L (2048, 64)
            # which codebook vector did we use for each feature (k_bins,num_enc_vectors)
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)  # 将匹配到的位置标记为1
            # k_bins, w: the sum of the encoder output, which are used with this codebook vector.
            _k_sum = t.matmul(x_l_onehot, x)    # 把x的值对应到码本的具体位置

            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins (2048), 每一个向量被使用的次数, 一次为0.01
            y = self._tile(x)   # x (64, 96) -> (2048, 96)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]    # 将y随机打乱

            # dist.broadcast(_k_rand, 0)
            # dist.all_reduce(_k_sum)
            # dist.all_reduce(_k_elem)

            # to perform the update all the tensor should be on the same device
            _k_rand = _k_rand.to(device)
            _k_sum = _k_sum.to(device)
            _k_elem = _k_elem.to(device)
            self.k = self.k.to(device)
            self.k_sum = self.k_sum.to(device)
            self.k_elem = self.k_elem.to(device)

            # Update centres
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins 
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float() # 把 k_elem 中大于1的元素标记出来, (2048)
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) \
                + (1 - usage) * _k_rand
            # x_l_onehot.mean(dim=-1)  # prob of each bin
            _k_prob = _k_elem / t.sum(_k_elem)
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-8)
                             )  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = t.sum(usage)
            dk = t.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy,
                    used_curr=used_curr,
                    usage=usage,
                    dk=dk)

    # 将x转换成合适的大小, 并对其求取范数
    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        if x.shape[-1] == self.emb_width:
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., :self.emb_width], x[..., self.emb_width:]
            prenorm = (t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape))) + \
                (t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape)))

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        # x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.t().to(device) # (96, 2048)
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0,
                                                                                      keepdim=True)  # (N * L, b), (64, 2048)
        min_distance, x_l = t.min(distance, dim=-1) # (min, min_indices)
        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        # N, width, T = x.shape
        x = x.to(device)
        N, T = x.shape

        # Preprocess
        x, prenorm = self.preprocess(x)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x) # x_l 向量的坐标
        x_d = self.dequantise(x_l)  # 找到向量

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        # commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)
        # q_latent_loss = F.mse_loss(x_d, x.detach())
        # e_latent_loss = F.mse_loss(x, x_d.detach())
        # commit_loss = q_latent_loss + self.mu * e_latent_loss
        commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)
        # q_latent_loss = F.mse_loss(x_d, x.detach())
        # e_latent_loss = F.mse_loss(x, x_d.detach())
        # commit_loss = q_latent_loss + self.mu * e_latent_loss


        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))
        return x_l, x_d, commit_loss, dict(fit=fit,
                                           pn=prenorm,
                                           **update_metrics)
