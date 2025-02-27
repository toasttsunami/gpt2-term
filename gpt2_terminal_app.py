import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt2_implementation import SimplifiedGPT2
from tqdm import tqdm

# Hyperparameters
VOCAB_SIZE = 256        # no of ascii characters
EMBED_DIM = 384         # the number of dimensions in the embedding vector of a token
NUM_HEADS = 6           # number of attention heads in a trans. layer
NUM_LAYERS = 6          # number of transfoemr layers
MAX_SEQ_LEN = 128       # the seq length passed into the transfoerm (context length)
BATCH_SIZE = 32         # number of sequences passed at a time
LEARNING_RATE = 3e-4    # lr for the optimizer
MAX_ITERS = 5000        #
EVAL_INTERVAL = 500     #
EVAL_ITERS = 200        #
DROPOUT = 0.2           #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

# Character-level tokenizer
def tokenize(text):
    return [ord(c) for c in text]

def detokenize(tokens):
    return "".join([chr(t) for t in tokens])

# Load data
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# Get random batches
def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# Training loop with evaluation
def train(model, train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for iter in tqdm(range(MAX_ITERS), desc="Training"):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"\nIter {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        
        xb, yb = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, MAX_SEQ_LEN)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Text generation
def generate_text(model, start_text, max_new_tokens=500):
    model.eval()
    tokens = tokenize(start_text)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(tokens[-MAX_SEQ_LEN:]).unsqueeze(0).to(device)
            logits, _ = model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
    return detokenize(tokens)

def main():
    text = load_data("tinyshakespeare.txt")
    data = torch.tensor(tokenize(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]
    
    model = SimplifiedGPT2(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)
    
    train(model, train_data, val_data)
    
    # Interactive loop
    while True:
        start_text = input("Enter some starting text (or 'quit'): ")
        if prompt.lower() == 'quit':
            break
        print("Generated:", generate_text(model, start_text))

if __name__ == "__main__":
    main()