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
            out_channels=self.hidden_size // self.num_attention_heads,  
            heads=self.num_attention_heads,
            concat=True, 
            dropout=config.attention_probs_dropout_prob
        )
        
        # Project GAT output back to hidden_size if needed
        self.gat_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Fusion layer
        self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fusion_dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def build_graph(self, x):
        """Convert sequence to fully-connected graph structure"""
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            seq_len = x.size(1)
        else:
            seq_len = x.size(0)
        
        # Create edges for fully connected graph
        edge_index = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, device=x.device).t()
        
        return edge_index

    def forward(self, hidden_states, transformer_output):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process each item in batch separately
        graph_outputs = []
        for item in hidden_states:
            # Build graph structure
            edge_index = self.build_graph(item)
            
            # Apply GAT layer
            graph_output = self.gat(item, edge_index)  
            
            # Project if needed
            graph_output = self.gat_projection(graph_output)
            
            graph_outputs.append(graph_output)
            
        # Stack back into batch [batch_size, seq_len, hidden_size]
        graph_output = torch.stack(graph_outputs)
        
        # Ensure shapes match for concatenation
        assert graph_output.shape == transformer_output.shape, \
            f"Shape mismatch: graph={graph_output.shape}, transformer={transformer_output.shape}"
        
        # Fuse transformer and graph outputs
        fused = torch.cat([transformer_output, graph_output], dim=-1)
        fused = self.fusion(fused)
        fused = self.fusion_dropout(fused)
        
        return fused 