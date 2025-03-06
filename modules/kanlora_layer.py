import math
import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig
from modules.kan_layer import KAN, KANLinear

class LoRAKANLinear(KANLinear):
    """LoRA-enhanced KAN linear layer with separate adapters for base and spline paths"""
    def __init__(self, lora_config: LoraConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        
        # Base weight LoRA parameters
        self.lora_base_A = nn.Parameter(torch.empty(
            lora_config.r, self.in_features))
        self.lora_base_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Spline weight LoRA parameters
        spline_dim = self.in_features * (self.grid_size + self.spline_order)
        self.lora_spline_A = nn.Parameter(torch.empty(
            lora_config.r, spline_dim))
        self.lora_spline_B = nn.Parameter(torch.empty(
            self.out_features, lora_config.r))
        
        # Initialize LoRA parameters
        self.reset_lora_parameters()
        self.freeze_original()

    def reset_lora_parameters(self):
        """Initialize LoRA while preserving original weights"""
        nn.init.kaiming_uniform_(self.lora_base_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_base_B)
        nn.init.kaiming_uniform_(self.lora_spline_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_spline_B)

    def freeze_original(self):
        """Freeze original KAN parameters"""
        self.base_weight.requires_grad_(False)
        self.spline_weight.requires_grad_(False)
        if self.enable_standalone_scale_spline:
            self.spline_scaler.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # Original KAN computations
        base_act = self.base_activation(x)
        base_out = F.linear(base_act, self.base_weight)
        spline_basis = self.b_splines(x).flatten(1)
        spline_out = F.linear(spline_basis, 
                            self.scaled_spline_weight.view(self.out_features, -1))
        
        # LoRA additions
        if self.lora_config.r > 0:
            # Base path LoRA
            lora_base = F.linear(base_act, self.lora_base_A)
            lora_base = F.linear(lora_base, self.lora_base_B)
            base_out += self.lora_config.lora_alpha/self.lora_config.r * lora_base
            
            # Spline path LoRA
            lora_spline = F.linear(spline_basis, self.lora_spline_A)
            lora_spline = F.linear(lora_spline, self.lora_spline_B)
            spline_out += self.lora_config.lora_alpha/self.lora_config.r * lora_spline
        
        return (base_out + spline_out).view(*x.shape[:-1], -1)

class LoRAKAN(KAN):
    """Wrapper for converting KAN layers to LoRA-KAN"""
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
            # Transfer original weights
            new_layer.load_state_dict(layer.state_dict(), strict=False)
            self.layers[i] = new_layer