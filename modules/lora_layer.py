# lora_layer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig


class LoRALinear(nn.Module):
    """
    LoRALinear layer using a LoRA configuration from the Hugging Face PEFT library.

    It wraps a frozen base linear transformation and adds a low-rank update:
      output = F.linear(x, weight, bias) + scaling * F.linear(F.linear(x, lora_A), lora_B)

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        lora_config (LoraConfig): A configuration object with attributes:
            - r: low-rank dimension.
            - alpha (or lora_alpha): scaling factor.
            - dropout: dropout probability for the LoRA path.
        bias (bool): Whether to include a bias term (default: True).
    """

    def __init__(self, in_features, out_features, lora_config: LoraConfig, bias=True):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Base (frozen) parameters.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA configuration parameters.
        self.lora_r = lora_config.r
        # Some LoraConfig objects name the scaling as either "alpha" or "lora_alpha"
        self.lora_alpha = getattr(lora_config, "lora_alpha", lora_config.alpha)
        self.scaling = self.lora_alpha / self.lora_r if self.lora_r > 0 else 1.0

        # LoRA adapter parameters (if enabled).
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.Tensor(self.lora_r, in_features))
            self.lora_B = nn.Parameter(torch.Tensor(out_features, self.lora_r))
        else:
            self.lora_A = None
            self.lora_B = None

        # Optional dropout for the LoRA branch.
        self.lora_dropout = nn.Dropout(p=lora_config.dropout) if lora_config.dropout > 0.0 else None

        self.reset_parameters()

        # Freeze the original weight and bias.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_parameters(self):
        # Initialize the frozen base weight and bias.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize LoRA adapter parameters.
        if self.lora_r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize lora_B to zeros so initially only the base weight contributes.
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Compute the frozen base output.
        result = F.linear(x, self.weight, self.bias)
        # If LoRA is enabled, compute and add the low-rank update.
        if self.lora_r > 0:
            x_input = self.lora_dropout(x) if self.lora_dropout is not None else x
            # Project input to low-rank space.
            lora_out = F.linear(x_input, self.lora_A)
            # Project back to output dimension.
            lora_out = F.linear(lora_out, self.lora_B)
            result = result + self.scaling * lora_out
        return result
