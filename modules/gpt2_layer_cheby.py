from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention
from modules.chebykan_layer import ChebyKANLayer

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention sub-layer.
        self.self_attention = CausalSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Use ChebyKANLayer if configured, otherwise use a feed-forward (FF) network.
        if getattr(config, "use_kan", False):
            print("Using ChebyKAN")
            self.kan_layer = ChebyKANLayer(config.hidden_size, config.intermediate_size, getattr(config, "kan_degree", 8))
            # If the ChebyKANLayer output dimension differs from hidden_size,
            # add a projection layer to map it back.
            if config.intermediate_size != config.hidden_size:
                self.kan_projection = nn.Linear(config.intermediate_size, config.hidden_size)
            else:
                self.kan_projection = None
        else:
            print("Using FF")
            # Standard feed-forward network components.
            self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
            self.interm_af = F.gelu
            self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(self, input, output, dense_layer, dropout):
        """
        Helper method to apply a dense transformation, dropout, 
        and subsequently add the transformed output back to the input.
        Note that this does not apply layer normalization.
        """
        output = dense_layer(output)
        output = dropout(output)
        return input + output

    def forward(self, hidden_states, attention_mask):
        # Apply pre-layer normalization before self-attention.
        norm_a = self.attention_layer_norm(hidden_states)
        attn_output = self.self_attention(norm_a, attention_mask)
        hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)
        
        # Apply pre-layer normalization before feed-forward or ChebyKAN network.
        norm_ff = self.out_layer_norm(hidden_states)
        if hasattr(self, "kan_layer"):
            batch_size, seq_len, hidden_dim = norm_ff.shape
            flat_norm = norm_ff.view(-1, hidden_dim)
            ff_output = self.kan_layer(flat_norm)
            # If the output dimension is different, project it back to hidden_dim.
            if self.kan_projection is not None:
                ff_output = self.kan_projection(ff_output)
            ff_output = ff_output.view(batch_size, seq_len, hidden_dim)
        else:
            ff_output = self.out_dense(self.interm_af(self.interm_dense(norm_ff)))
        
        hidden_states = self.add(hidden_states, ff_output, lambda x: x, self.out_dropout)
        return hidden_states
