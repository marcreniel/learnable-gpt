import math
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig
from modules.kan_layer import KAN, KANLinear

class DoRAKANLinear(KANLinear):
    """DoRA-enhanced KAN layer with base weight decomposition only"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        
        # Base path decomposition
        self.dora_magnitude = nn.Parameter(torch.ones(self.out_features))
        self.dora_A = nn.Parameter(torch.empty(
            lora_config.r, self.in_features))
        self.dora_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Initialize parameters
        self.reset_dora_parameters()
        self.freeze_original()

    def reset_dora_parameters(self):
        """Initialize DoRA components using KAN-aware initialization"""
        nn.init.kaiming_uniform_(self.dora_A, a=math.sqrt(5))
        nn.init.zeros_(self.dora_B)
        
        # Initialize magnitude from base weight norms
        with torch.no_grad():
            self.dora_magnitude.copy_(torch.norm(self.base_weight, dim=1))

    def freeze_original(self):
        """Freeze original KAN parameters"""
        self.base_weight.requires_grad_(False)
        self.spline_weight.requires_grad_(False)
        if self.enable_standalone_scale_spline:
            self.spline_scaler.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # Original base computations with decomposed weights
        base_act = self.base_activation(x)
        
        # Base path with DoRA decomposition
        base_direction = F.linear(base_act, self.base_weight)
        if self.lora_config.r > 0:
            base_lora = F.linear(F.linear(base_act, self.dora_A), self.dora_B)
            base_direction += self.lora_config.lora_alpha/self.lora_config.r * base_lora
        base_out = self.dora_magnitude * F.normalize(base_direction, dim=-1)
        
        # Original spline path (unchanged from KANLinear)
        spline_basis = self.b_splines(x).flatten(1)
        spline_out = F.linear(spline_basis, 
                            self.scaled_spline_weight.view(self.out_features, -1))
        
        return (base_out + spline_out).view(*x.shape[:-1], -1)

    def l1_regularization_loss(self):
        """Combined L1 regularization for base components only"""
        # Base path regularization
        base_reg = torch.sum(torch.abs(self.dora_magnitude)) + \
                 torch.sum(torch.abs(self.dora_A)) + \
                 torch.sum(torch.abs(self.dora_B))
        
        # Add original KAN regularization
        return super().regularization_loss() + 0.1 * base_reg

class DoRAKAN(KAN):
    """DoRA-adapted KAN implementation (base weights only)"""
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
