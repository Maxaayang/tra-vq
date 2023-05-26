import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
#     """
#     VQ-VAE layer: Input any tensor to be quantized. 
#     Args:
#         embedding_dim (int): the dimensionality of the tensors in the
#           quantized space. Inputs to the modules must be in this format as well.
#         num_embeddings (int): the number of vectors in the quantized space.
#         commitment_cost (float): scalar which controls the weighting of the loss terms (see
#           equation 4 in the paper - this variable is Beta).
#     """
    def __init__(self, embedding_dim, num_embeddings, beta):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = beta
        self.init = False
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
       
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        # print("x.shape ", x.shape)
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO 这里训练时要注释掉
        # device = torch.device('cuda:0')
        # print("x", x)
        # [B, H, W, C] -> [BHW, C]
        # x = x[:, :96, :]
        # if not self.init:
        #     self.init_emb(x)
        flat_x = x.reshape(-1, self.embedding_dim)
        # flat_x = flat_x.to(device)
        
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) # [B, H, W, C]

        # min_distance, x_l = self.get_code_indices(flat_x)
        # quantized = self.quantize(x_l)
        # quantized = quantized.view_as(x) # [B, H, W, C]
        
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # loss = torch.norm(quantized.detach() - x) ** 2 / np.prod(x.shape) * self.commitment_cost

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        # quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # quantized = quantized.permute(0, 2, 1).contiguous()
        # quantized = torch.squeeze(quantized)
        return encoding_indices, quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        # print("self.embeddings.weight ", self.embeddings.weight)
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices

        # distances = (
        #     torch.sum(flat_x ** 2, dim=-1, keepdim=True) +
        #     torch.sum(self.embeddings.t().to('cuda') ** 2, dim=0, keepdim=True) -
        #     2. * torch.matmul(flat_x, self.embeddings.t().to('cuda'))
        # ) # [N, M]

        # min_distance, x_l = torch.min(distances, dim=-1) # (min, min_indices)
        # # encoding_indices = torch.argmin(distances, dim=1) # [N,]
        # return min_distance, x_l
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 
        # x = F.embedding(encoding_indices, self.embeddings)
        # return x

