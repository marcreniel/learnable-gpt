from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention
from modules.chebykan_layer import ChebyKANLayer

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # KAN network (if used).
    if getattr(config, "use_kan", False):
      print("Using ChebyKAN")
      self.kan_layer = ChebyKANLayer(config.hidden_size, config.hidden_size, getattr(config, "kan_degree", 8))
    else:
      print("Using FF")
      # Feed forward.
      self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
      self.interm_af = F.gelu
      # Add-norm for feed forward.
      self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    ### YOUR CODE HERE
    # Apply dense layer, dropout, and add input
    output = dense_layer(output)
    output = dropout(output)
    output = input + output
    return output

  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    ### YOUR CODE HERE
    # Apply layer norm to hidden_states and pass through self-attention, then add dropout
    norm_a = self.attention_layer_norm(hidden_states)
    attn_output = self.self_attention(norm_a, attention_mask)
    hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)
    # Apply layer norm to hidden_states and pass through eithen KAN or ff layer, then add dropout and return
    norm_ff = self.out_layer_norm(hidden_states)
    if hasattr(self, "kan_layer"):
        ff_output = self.kan_layer(norm_ff)
    else:
        ff_output = self.out_dense(self.interm_af(self.interm_dense(norm_ff)))
    hidden_states = self.add(hidden_states, ff_output, lambda x: x, self.out_dropout)
    return hidden_states