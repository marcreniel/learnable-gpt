from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE

                # Get the hyperparameters
                b_1, b_2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Initialize the state if it is the first time
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # Update step count
                state["step"] += 1
                step = state["step"]

                # Get the exponential moving averages
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Update the exponential moving averages with the gradient
                exp_avg.mul_(b_1).add_(grad, alpha=1 - b_1)
                exp_avg_sq.mul_(b_2).addcmul_(grad, grad, value=1 - b_2)

                # Apply bias correction if necessary
                if correct_bias:
                    bias_correction1 = 1 - b_1 ** step
                    bias_correction2 = 1 - b_2 ** step
                    sqrt_bias_correction2 = math.sqrt(bias_correction2)
                    denominator = (exp_avg_sq.sqrt() / sqrt_bias_correction2).add_(eps)
                    step_size = alpha / bias_correction1
                else:
                    denominator = exp_avg_sq.sqrt().add_(eps)
                    step_size = alpha

                # Update the parameters
                p.data.addcdiv_(exp_avg, denominator, value=-step_size)

                # Apply weight decay if specified
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-weight_decay * alpha)

        return loss
