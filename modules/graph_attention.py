import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GraphAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Graph attention layer
        self.gat = GATConv(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            heads=self.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Fusion layer
        self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fusion_dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def build_graph(self, hidden_states):
        """Convert sequence to fully-connected graph structure"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create edges for fully connected graph (each token connected to all others)
        edge_index = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:  # Don't connect node to itself
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, device=hidden_states.device).t()
        
        return edge_index

    def forward(self, hidden_states, transformer_output):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process each item in batch separately since GAT expects 2D input
        graph_outputs = []
        for item in hidden_states:
            # Build graph structure
            edge_index = self.build_graph(item)
            
            # Apply GAT layer
            graph_output = self.gat(item, edge_index)
            graph_outputs.append(graph_output)
            
        # Stack back into batch
        graph_output = torch.stack(graph_outputs)
        
        # Fuse transformer and graph outputs
        fused = torch.cat([transformer_output, graph_output], dim=-1)
        fused = self.fusion(fused)
        fused = self.fusion_dropout(fused)
        
        return fused 