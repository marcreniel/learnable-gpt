import math
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig
from modules.kan_layer import KAN, KANLinear

class DoRAKANLinear(KANLinear):
    """DoRA-enhanced KAN layer with magnitude-direction decomposition"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        
        # Base path decomposition
        self.dora_base_magnitude = nn.Parameter(torch.ones(self.out_features))
        self.dora_base_A = nn.Parameter(torch.empty(
            lora_config.r, self.in_features))
        self.dora_base_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Spline path decomposition
        spline_dim = self.in_features * (self.grid_size + self.spline_order)
        self.dora_spline_magnitude = nn.Parameter(torch.ones(self.out_features))
        self.dora_spline_A = nn.Parameter(torch.empty(
            lora_config.r, spline_dim))
        self.dora_spline_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Initialize parameters
        self.reset_dora_parameters()
        self.freeze_original()

    def reset_dora_parameters(self):
        """Initialize DoRA components using KAN-aware initialization"""
        # Base path
        nn.init.kaiming_uniform_(self.dora_base_A, a=math.sqrt(5))
        nn.init.zeros_(self.dora_base_B)
        
        # Spline path
        nn.init.kaiming_uniform_(self.dora_spline_A, a=math.sqrt(5))
        nn.init.zeros_(self.dora_spline_B)
        
        # Magnitude parameters
        with torch.no_grad():
            self.dora_base_magnitude.copy_(torch.norm(self.base_weight, dim=1))
            spline_norms = torch.norm(self.spline_weight.view(
                self.out_features, -1), dim=1)
            self.dora_spline_magnitude.copy_(spline_norms)

    def freeze_original(self):
        """Freeze original KAN parameters"""
        self.base_weight.requires_grad_(False)
        self.spline_weight.requires_grad_(False)
        if self.enable_standalone_scale_spline:
            self.spline_scaler.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # Original KAN computations with decomposed weights
        base_act = self.base_activation(x)
        
        # Base path with DoRA decomposition
        base_direction = F.linear(base_act, self.base_weight)
        if self.lora_config.r > 0:
            base_lora = F.linear(F.linear(base_act, self.dora_base_A), self.dora_base_B)
            base_direction += self.lora_config.lora_alpha/self.lora_config.r * base_lora
        base_out = self.dora_base_magnitude * F.normalize(base_direction, dim=-1)
        
        # Spline path with DoRA decomposition and L1 regularization
        spline_basis = self.b_splines(x).flatten(1)
        spline_direction = F.linear(spline_basis, 
                                  self.scaled_spline_weight.view(self.out_features, -1))
        if self.lora_config.r > 0:
            spline_lora = F.linear(F.linear(spline_basis, self.dora_spline_A), 
                                 self.dora_spline_B)
            spline_direction += self.lora_config.lora_alpha/self.lora_config.r * spline_lora
        spline_out = self.dora_spline_magnitude * F.normalize(spline_direction, dim=-1)
        
        return (base_out + spline_out).view(*x.shape[:-1], -1)

    def l1_regularization_loss(self):
        """Combined L1 regularization for spline components"""
        # Spline path regularization (absolute values of magnitude + direction components)
        spline_reg = torch.sum(torch.abs(self.dora_spline_magnitude)) + \
                   torch.sum(torch.abs(self.dora_spline_A)) + \
                   torch.sum(torch.abs(self.dora_spline_B))
        
        # Add original KAN regularization
        return super().regularization_loss() + 0.1 * spline_reg

class DoRAKAN(KAN):
    """DoRA-adapted KAN implementation"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        self.convert_layers()

    def convert_layers(self):
        """Replace KANLinear layers with DoRAKANLinear"""
        for i, layer in enumerate(self.layers):
            new_layer = DoRAKANLinear(
                lora_config=self.lora_config,
                in_features=layer.in_features,
                out_features=layer.out_features,
                grid_size=layer.grid_size,
                spline_order=layer.spline_order,
                scale_noise=layer.scale_noise,
                scale_base=layer.scale_base,
                scale_spline=layer.scale_spline,
                enable_standalone_scale_spline=layer.enable_standalone_scale_spline,
                base_activation=layer.base_activation.__class__,
                grid_eps=layer.grid_eps,
            )
            new_layer.load_state_dict(layer.state_dict(), strict=False)
            self.layers[i] = new_layer

    def regularization_loss(self):
        """Aggregate regularization across all layers"""
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.l1_regularization_loss()
        return total_loss
