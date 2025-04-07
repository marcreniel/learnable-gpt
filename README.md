# Learnable-GPT: Testing Novel Frameworks on Multitask Fine-Tuned GPT-2 with LoRA

Learnable-GPT is a research project aimed at improving GPT-based transformer models through the integration of innovative adaptive learning techniques. This repository implements several architectural enhancements including Kolmogorov-Arnold Networks (KAN), Low-Rank Adaptation (LoRA), and Graph Attention mechanisms to create more efficient and powerful language models.

## Project Overview

This research explores how transformer architectures can be enhanced through specialized adaptations that improve model efficiency and performance.

## Technical Background

### Kolmogorov-Arnold Networks (KAN)

KANs are neural networks based on the Kolmogorov-Arnold representation theorem, which states that any continuous multivariate function can be represented as a composition of continuous functions of a single variable and addition[3]. In practice, KANs often provide better expressivity than traditional dense networks with fewer parameters.

### Low-Rank Adaptation (LoRA)

LoRA enables efficient fine-tuning of large pre-trained models by injecting trainable low-rank matrices into each layer of the Transformer architecture[3]. This dramatically reduces the number of trainable parameters while maintaining model performance.

### Graph Attention Networks

Graph attention mechanisms allow the model to consider structural relationships between tokens beyond the sequential nature of traditional transformers[3]. By modeling the input as a fully-connected graph, the model can attend to semantically related tokens regardless of their position in the sequence.

## Core Components

- **Enhanced GPT2Layer**: A modified GPT-2 layer implementation with support for multiple adaptation techniques
- **Flexible Module Selection**: Easily switch between standard feed-forward networks, KAN, LoRA, or combined approaches
- **Graph Attention Integration**: Optional graph-based attention mechanism that complements traditional self-attention
- **Parameter-Efficient Fine-tuning**: LoRA implementations that reduce the number of trainable parameters

## Key Modules

The repository contains several specialized modules:

1. **CausalSelfAttention**: Enhanced self-attention implementation for causal language modeling
2. **KAN Layer**: Implementation of Kolmogorov-Arnold Networks for improved function approximation
3. **LoRALinear**: Linear layers with Low-Rank Adaptation capabilities
4. **LoRAKAN**: Combined implementation of LoRA and KAN for enhanced adaptation
5. **GraphAttentionLayer**: Graph-based attention mechanism using PyTorch Geometric's GATConv

## Installation

```bash
# Clone the repository
git clone https://github.com/marcreniel/learnable-gpt.git
cd learnable-gpt

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)

## Usage

### Basic Model Configuration

```python
from model import GPT2Config, GPT2LMHeadModel

# Configure a model with KAN layers
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    use_kan=True
)

# Initialize the model
model = GPT2LMHeadModel(config)
```

### Using LoRA Adaptation

```python
from peft import LoraConfig
from model import GPT2Config, GPT2LMHeadModel

# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,
    alpha=16,
    dropout=0.1
)

# Create model config with LoRA enabled
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    use_lora=True,
    lora_config=lora_config
)

# Initialize the model
model = GPT2LMHeadModel(config)
```

### Enabling Graph Attention

```python
from model import GPT2Config, GPT2LMHeadModel

# Create model config with graph attention enabled
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    use_graph=True
)

# Initialize the model
model = GPT2LMHeadModel(config)
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `use_kan` | Enable Kolmogorov-Arnold Networks | False |
| `use_lora` | Enable Low-Rank Adaptation | False |
| `use_graph` | Enable Graph Attention mechanisms | False |
| `lora_config` | LoRA configuration object | None |
| `hidden_size` | Dimension of hidden representations | 768 |
| `intermediate_size` | Dimension of intermediate feed-forward layer | 3072 |
| `hidden_dropout_prob` | Dropout probability for hidden layers | 0.1 |
| `hidden_hybrid_dropout_prob` | Dropout probability for hybrid networks | 0.1 |

## Experimental Results

Our experiments reveal that optimized LoRA-enhanced transformers consistently outperform more complex architectures like KAN-LoRA and Graph-LoRA. Key findings include:

- LoRA fine-tuning of GPT2-Large achieved the strongest overall performance
- KAN layers showed promise on complex problems like sonnet generation
- LoRA's strength comes from combining weight decay and dropout on low-rank update parameters
- The additional complexity of graph-based or spline-based modules can hamper training convergence

## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments

We want to express our sincere gratitude to the Stanford CS224N: Natural Language Processing with Deep Learning course staff for their guidance and support throughout this research project. The course infrastructure, resources, and feedback were instrumental in enabling us to explore the integration of novel techniques such as Kolmogorov-Arnold Networks (KAN), Graph Attention Networks (GAT), and Low-Rank Adaptation (LoRA) within transformer-based language models.

Special thanks to the authors of this research paper:

- **Marc Bernardino**: Developed the base model, implemented KAN and KAN-LORA, set up Wandb logging, created figures, researched, and edited the paper.
- **Gabriel Bo**: Implemented LoRA with KAN and GAT, trained models on GPU, assisted with the base model, performed research, and wrote the report.
- **Justin Gu**: Built the graph attention architecture, supported the base model, managed GPU training, wrote the approach section, and conducted research.

We also acknowledge the authors of the foundational research that our work builds upon, particularly those working on Kolmogorov-Arnold Networks, Low-Rank Adaptation techniques, and Graph Attention Networks.

## Citation

If you use Learnable-GPT in your research, please cite our work:

```
@misc{2025learnablegpt,
  author = {Bernardino, Marc and Bo, Gabriel and Gu, Justin},
  title = {Learnable-GPT: Enhancing Transformer Models with Adaptive Learning Techniques},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/marcreniel/learnable-gpt}}
}
```
