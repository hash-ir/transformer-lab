"""
Components of the Transformer model.
References:
1) The nanoGPT implementation by Andrej Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py 
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Head(nn.Module):
    """Single self-attention head."""

    def __init__(self, n_embed, block_size, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x) # query
        k = self.key(x) # key
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # value
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention.""" 

    def __init__(self, n_head, n_embed, block_size, head_size, dropout=0.2) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_embed (=head_size * num_heads))
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class SelfAttention(nn.Module):
    """
    Similar to `MultiHeadAttention` but applies 
    multiple heads in parallel.
    """
    def __init__(self, n_head, n_embed, block_size, dropout=0.2) -> None:
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # output projection
        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = nn.Dropout(dropout)
        # causal mask to ensure that attention is only 
        # applied to the left in the input sequence 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
     
    def forward(self, x):
        # batch size, sequence length (block_size), embedding dim (n_embed)
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch 
        # and move head forward to be the batch dim
        q, k, v = self.attn(x).split(self.n_embed, dim=-1)
        # reshape q, k, v -> (batch size, n_head, block_size, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ v
        # re-assemble all head outputs side by side
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.droout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """Two-layer feed-forward network."""

    def __init__(self, n_embed, dropout=0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    """Implemention of LayerNorm: https://arxiv.org/abs/1607.06450."""

    def __init__(self, dim, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / (torch.sqrt(xvar + self.eps))
        x = self.gamma * xhat + self.beta
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation.""" 

    def __init__(self, block_size, n_embed, n_head) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embed)
        self.sa = SelfAttention(n_head, n_embed, block_size)
        self.ln2 = LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Decoder(nn.Module):
    """
    The decoder block of Transformer,
    based on https://arxiv.org/pdf/1706.03762.
    """
    def __init__(self, vocab_size, block_size, n_head, n_embed, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # use multi-head attention blocks
        self.blocks = nn.Sequential(*(Block(block_size, n_embed, n_head=n_head) for _ in range(n_layer)))
        self.ln_f = LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        # x = self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        # x = self.ffwd(x) # go through 1-layer feed-forward
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # idx_next = torch.argmax(probs, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    