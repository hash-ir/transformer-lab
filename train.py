import torch
from models import BigramLanguageModel

# data and model configuration 
split = 0.9
batch_size = 32  
block_size = 8 
max_iters = 5000
eval_iters = 200  
eval_interval = 500 
learning_rate = 1e-3
n_embed = 32
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

def get_batch(split):
    """Get batched data based on the split i.e 
    training or validation

    Args:
        split (str): train or validation

    Returns:
        tuple: text input and labels
    """    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y


@torch.no_grad()
def estimate_loss():
    """Compute loss for training and validation
    data

    Returns:
        dict: losses (training and validation)
    """    
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


model = BigramLanguageModel(vocab_size, block_size, n_embed, n_embed)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
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