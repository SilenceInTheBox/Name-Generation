import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16
block_size = 8
max_iters = 100
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 16
n_head = 4
n_layer = 3
dropout = 0.0
# ------------

torch.manual_seed(1337)

with open(r'./input.txt', 'r', encoding='uft-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
def encode(x): return [stoi[i] for i in x]
def decode(x): return ''.join([itos[i] for i in x])


data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones((block_size, block_size))))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = q @ k.transpose(-1, -2) * C**-0.5
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.key = torch.randn((1, num_heads, n_embd, head_size))
        self.query = torch.randn((1, num_heads, n_embd, head_size))
        self.value = torch.randn((1, num_heads, n_embd, head_size))
        self.register_buffer('tril', torch.tril(
            torch.ones((block_size, block_size))))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''WATCH OUT! output shape has to be revised!'''
        B, T, C = x.shape
        x = x.view((B, 1, T, C))
        k = x @ self.key
        q = x @ self.query
        v = x @ self.value

        wei = q @ k.transpose(-1, -2) * C**-0.5
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        out = out.permute(0, 2, 1, 3).reshape((B, T, -1))
        return out

# TODO: FFWD Module, LayerNorm Module, Final Module, Optimizer
