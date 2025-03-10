from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention
from modules.kan_layer import KAN 
from peft import LoraConfig
from modules.graph_attention import GraphAttentionLayer

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        # KAN network (if enabled).
        if getattr(config, "use_kan", False) and getattr(config, "use_lora", False):
            from modules.lorakan_layer import LoRAKAN
            # LoRAKAN-MLP network.
            print("Using LoRA-KAN-NLP network")
            self.interm_kan = LoRAKAN(layers_hidden=[config.hidden_size, int(config.hidden_size*2)], lora_config=config.lora_config)
            self.interm_af = F.gelu
            self.out_kan = LoRAKAN(layers_hidden=[int(config.hidden_size*2), config.hidden_size], lora_config=config.lora_config)
        elif getattr(config, "use_kan", False):
            # KAN-MLP network.
            print("Using KAN-MLP network")
            self.interm_kan = KAN(layers_hidden=[config.hidden_size, config.intermediate_size])
            self.interm_af = F.gelu
            self.interm_dropout = nn.Dropout(config.hidden_hybrid_dropout_prob)
            self.out_kan = KAN(layers_hidden=[config.intermediate_size, config.hidden_size])
        # LoRA network (if enabled).
        elif getattr(config, "use_lora", False):
            print("Using LoRA network")
            from modules.lora_layer import LoRALinear
            self.interm_dense = LoRALinear(config.hidden_size, config.intermediate_size, lora_config=config.lora_config)
            self.interm_af = F.gelu
            self.out_dense = LoRALinear(config.intermediate_size, config.hidden_size, lora_config=config.lora_config)
        else:
            # Feed forward block (Base Model).
            self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
            self.interm_af = F.gelu
            self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
            
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Add graph attention if enabled
        self.use_graph = getattr(config, "use_graph", False)
        if self.use_graph:
            self.graph_attention = GraphAttentionLayer(config)
            self.graph_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def add(self, input, output, dense_layer, dropout):
        """
        Helper method:
        - Applies a dense layer followed by dropout,
        - And adds the result back to the input (residual connection).
        """
        output = dense_layer(output)
        output = dropout(output)
        return input + output

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass of the GPT-2 layer.
        - First applies multi-head self-attention with pre-layer normalization.
        - Then applies either the KAN network or a feed-forward network.
        - If graph attention is enabled, applies it after the feed-forward network.
        - Residual connections are used after each sub-layer.
        """
        # Multi-head self-attention sub-layer.
        norm_a = self.attention_layer_norm(hidden_states)
        attn_output = self.self_attention(norm_a, attention_mask)
        hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)
        
        # Feed-forward or KAN sub-layer.
        norm_ff = self.out_layer_norm(hidden_states)
        if hasattr(self, "interm_kan"):
            # For KAN, flatten the sequence dimensions so that the input is 2D.
            batch_size, seq_len, hidden_dim = norm_ff.shape
            flat_norm = norm_ff.view(-1, hidden_dim)
            ff_output = self.out_kan(self.interm_af(self.interm_kan(flat_norm)))
            ff_output = ff_output.view(batch_size, seq_len, hidden_dim)
        else:
            ff_output = self.out_dense(self.interm_af(self.interm_dense(norm_ff)))
            
        hidden_states = self.add(hidden_states, ff_output, lambda x: x, self.out_dropout)

        # Apply graph attention if enabled
        if self.use_graph:
            norm_g = self.graph_layer_norm(hidden_states)
            graph_output = self.graph_attention(norm_g, hidden_states)
            hidden_states = self.add(hidden_states, graph_output, lambda x: x, self.out_dropout)

        return hidden_states
