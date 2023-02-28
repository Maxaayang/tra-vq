import torch
import torch.nn as nn



def rbf_prob(dist, smooth):
    prob = torch.exp(-torch.multiply(dist, 0.5*smooth))
    probs = prob/torch.unsqueeze(torch.sum(prob, 1),1)
    return probs

# class OhVectorQuantizer():
#   # b: batch size; q: number of channels; K: number of codewords; d:embedding_dim; 
#   def __init__(self, embedding_dim, num_embeddings, commitment_cost, name='vq_layer'):
#     self._embedding_dim = embedding_dim
#     self._num_embeddings = num_embeddings
#     self._commitment_cost = commitment_cost

#     initializer = tf.initializers.variance_scaling()
#     self._w = tf.get_variable('embedding', [self._embedding_dim, self._num_embeddings], initializer=initializer, trainable=True)
  

# class OhVectorQuantizer(nn.Module):
#   def __init__(self, embedding_dim, num_embeddings, commitment_cost, name='vq_layer'):
#         super(OhVectorQuantizer, self).__init__()
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
#         self._commitment_cost = commitment_cost

#         initializer = torch.nn.init.xavier_uniform_
#         self._w = nn.Parameter(initializer(torch.empty(self._embedding_dim, self._num_embeddings)))
#         self._w.requiresGrad = True

#   def forward(self, inputs, is_training):
#         #noisy
#         #inputs['z_mean'] = add_noise(inputs['z_mean'], 0.01)
#         # Assert last dimension is same as self._embedding_dim
#         w = self._w
      
#         # shape: [batch, num_channel, embedding_dim]
#         # input_shape = torch.shape(inputs['z_mean'])
#         # with tf.control_dependencies([
#         #     tf.Assert(torch.equal(input_shape[-1], self._embedding_dim),[input_shape])]):
#         flat_inputs = torch.reshape(inputs['z_mean'], [-1, self._embedding_dim])
#         flat_smooth = torch.reshape(inputs['z_log_var'], [-1, self._num_embeddings])

#          # distances dimension: (b*q)*K
#         distances = (torch.sum(flat_inputs**2, 1, keepdims=True)
#                      - 2 * torch.matmul(flat_inputs, w)
#                      + torch.sum(w ** 2, 0, keepdims=True))
        
#         #after shape: (b*q)*K
#         smooth = 1./torch.exp(flat_smooth)**2
#         probs = rbf_prob(distances, smooth)/torch.sqrt(smooth)
#         #After shape: (q*b,1,K)
#         probs = torch.unsqueeze(probs, 1)
#         #After shape: (1,d,K)
#         codebook = torch.unsqueeze(w, 0)
#         #expected shape: b*q*d
#         quantize_vector = torch.sum(codebook*probs,2)
#         quantized = torch.reshape(quantize_vector, inputs['z_mean'].shape)
    
#         #encoding_indices = tf.argmax(- distances, 1)
#         #values dimension: flat*2
#         #[values, encoding_indices] = tf.nn.top_k(-distances, k = 2)
#         #encoding_indices = tf.reshape(encoding_indices[:,0], input_shape[:-1])
#         #quantized = self.quantize(encoding_indices)

        
#         e_latent_loss = torch.mean((quantized.detach() - inputs['z_mean']) ** 2)
#         q_latent_loss = torch.mean((quantized - inputs['z_mean'].detach()) ** 2)
#         loss = q_latent_loss + self._commitment_cost * e_latent_loss 

#         quantized = inputs['z_mean'] + (quantized - inputs['z_mean']).detach()
        
#         return {'quantize': quantized, 'loss': loss}
    

import torch
import torch.nn as nn


class OhVectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, name='vq_layer'):
        super(OhVectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        initializer = torch.nn.init.xavier_uniform_
        self._w = nn.Parameter(initializer(torch.empty(self._embedding_dim, self._num_embeddings)))

    def forward(self, inputs, is_training):
        # noisy
        # inputs['z_mean'] = add_noise(inputs['z_mean'], 0.01)

        # Assert last dimension is same as self._embedding_dim
        w = self._w

        # shape: [batch, num_channel, embedding_dim]
        input_shape = inputs['z_mean'].shape
        assert input_shape[-1] == self._embedding_dim

        flat_inputs = inputs['z_mean'].reshape(-1, self._embedding_dim)
        flat_smooth = inputs['z_log_var'].reshape(-1, self._num_embeddings)

        # distances dimension: (b*q)*K
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                     - 2 * torch.matmul(flat_inputs, w)
                     + torch.sum(w ** 2, dim=0, keepdim=True))

        # after shape: (b*q)*K
        smooth = 1./torch.exp(flat_smooth)**2
        probs = rbf_prob(distances, smooth)/torch.sqrt(smooth)

        # After shape: (q*b,1,K)
        probs = probs.unsqueeze(1)

        # After shape: (1,d,K)
        codebook = w.unsqueeze(0)

        # expected shape: b*q*d
        quantize_vector = torch.sum(codebook*probs, 2)
        quantized = quantize_vector.reshape(inputs['z_mean'].shape)

        # encoding_indices = torch.argmax(-distances, dim=1)
        # values dimension: flat*2
        # [values, encoding_indices] = torch.topk(-distances, k=2)
        # encoding_indices = encoding_indices[:, 0].reshape(input_shape[:-1])
        # quantized = self.quantize(encoding_indices)

        e_latent_loss = torch.mean((quantized.detach() - inputs['z_mean']) ** 2)
        q_latent_loss = torch.mean((quantized - inputs['z_mean'].detach()) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs['z_mean'] + (quantized - inputs['z_mean']).detach()

        return {'quantize': quantized, 'loss': loss}


#   def embeddings(self):
#         return self._w
  
#   def quantize(self, encoding_indices):
#         with tf.control_dependencies([encoding_indices]):
#             w = tf.transpose(self.embeddings.read_value(), [1, 0])
#         return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)