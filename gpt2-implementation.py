import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        attn = F.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5), dim=-1)
        output = attn @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, embed_dim // num_heads) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        return self.output_linear(
            torch.cat([h(x) for h in self.heads], dim=-1)
        )

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class SimplifiedGPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)
        
        return x
