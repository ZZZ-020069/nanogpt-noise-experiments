"""GPT blocks adapted from nanoGPT (see third_party/nanogpt-LICENSE)."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shape:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304


class LayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3*c.n_embd, bias=False)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.heads = c.n_head

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [a.reshape(b,t,self.heads,d//self.heads).transpose(1,2)
                   for a in self.c_attn(x).chunk(3,dim=-1)]
        y = F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
        return self.c_proj(y.transpose(1,2).contiguous().reshape(b,t,d))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd,4*c.n_embd,bias=False)
        self.c_proj = nn.Linear(4*c.n_embd,c.n_embd,bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.ln_2 = LayerNorm(c.n_embd), LayerNorm(c.n_embd)
        self.attn, self.mlp = Attention(c), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c=Shape()):
        super().__init__()
        self.shape = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.n_embd),
            wpe=nn.Embedding(c.block_size,c.n_embd),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=LayerNorm(c.n_embd)))
        self.lm_head = nn.Linear(c.n_embd,c.vocab_size,bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        # Initialize each unique parameter once, independently of trainable scope.
        for name, p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p)
            else:
                std = .02 / math.sqrt(2*c.n_layer) if name.endswith('c_proj.weight') else .02
                nn.init.normal_(p, mean=0., std=std)

    def forward(self, ids, targets):
        pos = torch.arange(ids.shape[1],device=ids.device)
        x = self.transformer.wte(ids) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        return F.cross_entropy(logits.flatten(0,1),targets.flatten())


def set_scope(model, scope):
    if scope == 'full':
        k = model.shape.n_layer
    elif scope.startswith('last'):
        k = int(scope[4:])
        if not 1 <= k <= model.shape.n_layer:
            raise ValueError('scope exceeds model depth')
    else:
        raise ValueError('unknown scope')
    for name, p in model.named_parameters():
        active = scope == 'full' or (name.startswith('transformer.h.')
                   and int(name.split('.')[2]) >= model.shape.n_layer-k)
        p.requires_grad_(active)
    return [(n,p) for n,p in model.named_parameters() if p.requires_grad]
