from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq length
    vocab_size: int = 50257 # number of tokens 
    n_layer: int = 12 # number layesr
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension