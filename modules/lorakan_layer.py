import math
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig
from modules.kan_layer import KAN, KANLinear

class LoRAKANLinear(KANLinear):
    """LoRA-enhanced KAN layer with low-rank adaptation"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        
        # LoRA components
        self.lora_A = nn.Parameter(torch.empty(
            lora_config.r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Initialize parameters
        self.reset_lora_parameters()
        self.freeze_original()

    def reset_lora_parameters(self):
        """Initialize LoRA components"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_original(self):
        """Freeze original KAN parameters"""
        self.base_weight.requires_grad_(False)
        self.spline_weight.requires_grad_(False)
        if self.enable_standalone_scale_spline:
            self.spline_scaler.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # Original base computations
        base_act = self.base_activation(x)
        
        # Base path with LoRA
        base_out = F.linear(base_act, self.base_weight)
        if self.lora_config.r > 0:
            lora_adjustment = F.linear(F.linear(base_act, self.lora_A), self.lora_B)
            base_out += self.lora_config.lora_alpha/self.lora_config.r * lora_adjustment
        
        # Original spline path
        spline_basis = self.b_splines(x).flatten(1)
        spline_out = F.linear(spline_basis, 
                            self.scaled_spline_weight.view(self.out_features, -1))
        
        return (base_out + spline_out).view(*x.shape[:-1], -1)

    def l1_regularization_loss(self):
        """Combined L1 regularization for LoRA components"""
        lora_reg = torch.sum(torch.abs(self.lora_A)) + \
                 torch.sum(torch.abs(self.lora_B))
        
        # Add original KAN regularization
        return super().regularization_loss() + 0.1 * lora_reg

class LoRAKAN(KAN):
    """LoRA-adapted KAN implementation"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        self.convert_layers()

    def convert_layers(self):
        """Replace KANLinear layers with LoRAKANLinear"""
        for i, layer in enumerate(self.layers):
            new_layer = LoRAKANLinear(
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