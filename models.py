import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, n_embed, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.key(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        if T != 8:
            print(T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel""" 

    def __init__(self, n_heads, n_embed, block_size, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_embed (=head_size * num_heads))



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, head_size, n_embed):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed, block_size, head_size)
        self.sa_heads = MultiHeadAttention(4, n_embed, block_size, n_embed//4) # i.e. 4 heads  of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

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