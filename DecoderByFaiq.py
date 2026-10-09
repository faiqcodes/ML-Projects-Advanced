import torch
import urllib.request
import torch.nn as nn 
import torch.nn.functional as F


url =  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url , "input.txt")
# Updated Hyperparameters

block_size = 64
d_model    = 128
n_heads     = 4
batch_size = 32
head_size  = d_model // n_heads
n_layer    = 4
dropout    = 0.2

# Training settings
eval_interval = 300
max_iters    = 10000  
learning_rate = 1e-3  
device       = 'cuda' if torch.cuda.is_available() else 'cpu'



with open("input.txt" , "r" , encoding="utf-8") as f:
    text = f.read()
print(f"Lenght of Dataset in Character:{len(text)}")
print(text[:200])

print("="*35)
print("Now Tokenizer")
print("="*35)
chars = sorted(list(set(text)))

stoi = {ch : i  for i, ch in  enumerate(chars)}
itos = {i  : ch for i, ch in  enumerate(chars)}

encode = lambda s :       [stoi[c] for c in s ]
decode = lambda y : "".join(itos[x] for x in y) 

print(encode("hi there"))
print(decode(encode("hi there")))

data = torch.tensor(encode(text) , dtype=torch.long)
print(f"Shape of Encoded Data is:{data.shape} \n datatype {data.dtype} ")
print(data[:100])

print("="*35)
print("Splitting Data")
print("="*35)

n = int(0.9 *len(data))
train_data = data[:n]
val_data   = data[n:]

print("="*35)
print("Token Embedding")
print("="*35)

vocab_size = len(chars)

token_embedding = nn.Embedding(vocab_size , d_model)
pos_embedding   = nn.Embedding(block_size , d_model)

x = train_data[:block_size].unsqueeze(0)   
out = token_embedding(x)

B , T = x.shape
tok_emb = token_embedding(x)
pos = torch.arange(T)

pos_emb = pos_embedding(pos) 
x_embedded = tok_emb + pos_emb 

print("Token Embedding shape:", tok_emb.shape)
print("Position Embedding shape:", pos_emb.shape)
print("Final Output shape:", x_embedded.shape)
print(x.shape)
print(out.shape)
print("Vocab size length :",vocab_size)

print("="*35)
print("Decoder Block * N")
print("="*35)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key     = nn.Linear(d_model , head_size , bias = False)
        self.query   = nn.Linear(d_model , head_size , bias = False)
        self.value   = nn.Linear(d_model , head_size , bias = False)
        self.register_buffer("tril" , torch.tril(torch.ones(block_size , block_size)))
        self.dropout = nn.Dropout(dropout) 

    def forward(self , x ):
        B, T, C = x.shape
        k       = self.key(x)
        q       = self.query(x)
        v       = self.value(x)

        wei = q @ k.transpose(-2 , -1) * (head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei , dim=-1)
        wei = self.dropout(wei)

        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self , n_heads , head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj    = nn.Linear(d_model , d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        head_size = d_model // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layer)])
final_ln = nn.LayerNorm(d_model)

transformer_out = final_ln(blocks(x_embedded))

print("Complete Transformer Architecture Ready!")
print("Input Shape:", x_embedded.shape)
print("Transformer Output Shape:", transformer_out.shape)

print("="*35)
print("Step 05 : Final Norm + Linear Head")
print("="*35)

ln_f    = nn.LayerNorm(d_model)
lm_head = nn.Linear(d_model , vocab_size)
x_norm  = ln_f(out)
logits  = lm_head(x_norm)
print("Logits Shape:", logits.shape)

print("="*35)
print("Step 06 : Softmax + Sampling")
print("="*35)

last_token_logtis = logits[: , -1 ,:]
probs             = F.softmax(last_token_logtis , dim=-1)
next_token_id     = torch.multinomial(probs , num_samples=1)
next_char         = decode(next_token_id[0].tolist())

print("Probabilities Shape:", probs.shape)
print("Predicted Next Token ID:", next_token_id.item())
print("Predicted Next Character:", repr(next_char))

print("="*35)
print("Training Decoder And Wrapping :")
print("="*35)


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_tabel    = nn.Embedding(vocab_size , d_model)
        self.positon_embdedding_table = nn.Embedding(block_size , d_model)
        self.blocks                   = nn.Sequential(*[Block(d_model , n_heads) for _ in range(n_layer)])
        self.ln_f                     = nn.LayerNorm(d_model)
        self.lm_head                  = nn.Linear(d_model , vocab_size)

    def forward(self , idx , targets = None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_tabel(idx)
        pos_emd = self.positon_embdedding_table(torch.arange(T , device=idx.device))
        x       = tok_emb + pos_emd
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)
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
            idx_cond  = idx[:, -block_size:] # Context ko crop karna
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] # Last time-step ke logits
            probs     = F.softmax(logits, dim=-1) # Step 6: Softmax
            idx_next  = torch.multinomial(probs, num_samples=1) # Step 6: Sampling
            idx       = torch.cat((idx, idx_next), dim=1) # Next token append karna
        return idx

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i+block_size] for i in ix])
    y    = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("="*35)
print("Starting Training Loop...")
print("="*35)

for iter in range(max_iters):
    # Train batch get karna
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass & Optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Progress Print
    if iter % eval_interval == 0 or iter == max_iters - 1:
        print(f"Step {iter}/{max_iters} | Loss: {loss.item():.4f}")

print("="*35)
print("Generated Shakespeare Text After Training:")
print("="*35)

context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with newline
generated_tokens = model.generate(context, max_new_tokens=300)[0].tolist()
print(decode(generated_tokens))