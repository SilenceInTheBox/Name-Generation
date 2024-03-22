import torch
import torch.nn as nn
from torch.nn import functional as F

# last steps need to be taken for a fully working transformer. merge the modules below to a working model.
# further, we need to implement some more feature such as layernorm and residual connections.

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

with open(r'text_input/input.txt', 'r', encoding='utf-8') as file:
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


# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


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
        '''WATCH OUT! output shape has to be revised!
        -> seems alright...'''
        B, T, C = x.shape
        x = x.view((B, 1, T, C))
        k = x @ self.key
        q = x @ self.query
        v = x @ self.value
        # k,q,v: (B,nH,T,sH)

        wei = q @ k.transpose(-1, -2) * C**-0.5
        # wei: (B,nH,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # wei: (B,nH,T,T)

        out = wei @ v
        # out: (B,nH,T,sH)
        out = out.transpose(1, 2).reshape((B, T, -1))
        # out: (B,T,nH,sH) -> (B,T,nH*sH) = (B,T,C)
        return out

# TODO: FFWD Module, LayerNorm Module, Final Module, Optimizer


class Block(nn.Module):
    # TODO: add layernorm, residual connection, and dropout
    def __init__(self, n_emb, n_heads):
        super().__init__()
        self.linlayer = nn.Linear(n_emb, n_emb)
        self.attention = MultiHeadAttention(n_heads, n_emb // n_heads)
        self.layernorm = nn.LayerNorm(n_embd)

    def forward(self, x):
        res = self.linlayer(x)
        res = self.attention(res)
        res = self.layernorm(res)
        x = x + res
        return x


class Predictor(nn.Module):
    # TODO: add generation method. think about how the loss is evaluated
    def __init__(self, n_emb):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_emb)
        self.block1 = Block(n_emb, 4)
        self.block2 = Block(n_emb, 4)
        self.linear = nn.Linear(n_emb, vocab_size)

    def forward(self, x, y=None):   # (B,T) -> (B,T,n_emb) -> machine -> (B,T,vocab_size)
        emb = self.emb(x)
        x = self.block1(emb)
        x = self.block2(x)
        logits = self.linear(x)

        loss = None
        if y != None:
            logits = logits.view(-1, vocab_size)
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(x[:, -8:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            new_x = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, new_x), dim=1)
        return x


if __name__ == '__main__':
    print('<CASUAL DEBUG>')
    print(f'CUDA active: {torch.cuda.is_available()}\n')

    model = Predictor(n_embd)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_steps = 5000
    for i in range(1, max_steps+1):
        xtr, ytr = get_batch('train')
        optimizer.zero_grad()
        logits, loss = model(xtr, ytr)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f'step {i}/{max_steps}: {loss.item():.3f}')

    x = torch.zeros((1, 1), dtype=torch.long)
    test = model.generate(x, 250)

    print(''.join(decode(test[0].tolist())))
