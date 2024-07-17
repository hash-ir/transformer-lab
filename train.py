import torch
import torch.nn as nn
from models import Decoder

# data and model configuration 
split = 0.9
batch_size = 64 
block_size = 256
max_iters = 5000
eval_iters = 200  
eval_interval = 500 
learning_rate = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[c] for c in s])

data = torch.tensor(encode(text), dtype=torch.long)
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
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
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
    for split in ['train', 'val']:
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
        print(f"step {i}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))