import os
import requests

import torch
import torch.nn as nn

from models import Decoder
from tokenizer import encode, decode

# data and model configuration
split = 0.9
batch_size = 64
block_size = 128
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
n_embed = 64
n_head = 2
n_layer = 6
dropout = 0.2
num_merges = 20  # number of merges of multiple byte-pairs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

# Get data
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    print("Downloading tiny shakespeare...")
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    text = f.read()

# encode text to tokens and prepare training data
tokens, merges = encode(text, return_merge_dict=True)
vocab_size = 256 + num_merges  # 0..255 unicode tokens plus additional merged tokens
data = torch.tensor(tokens, dtype=torch.long)
n = int(split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(train_mode: bool):

    """Get batched data based on the split i.e
    training or validation

    Args:
        split (str): train or validation

    Returns:
        tuple: text input and labels
    """
    data = train_data if train_mode else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(m: nn.Module):
    """Compute loss for training and validation
    data

    Returns:
        dict: losses (training and validation)
    """
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


model = Decoder(vocab_size, block_size, n_head, n_embed, n_layer)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {i}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), merges=merges))
