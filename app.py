import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import string
import numpy as np
import os

# ------------------------------
# Load vocab
# ------------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

max_len = 100

# ------------------------------
# Tokenization + Encoding
# ------------------------------
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip().split()

def encode_tokens(tokens):
    tokens = tokens[:max_len]
    ids = [word2idx.get(token, 1) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

# ------------------------------
# Model classes
# ------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        B, T, E = x.size()
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(B, T, E)
        return self.fc_out(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128,
                 num_classes=5, num_layers=2, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))   # [CLS] token
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))  # +1 for CLS

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T = x.size()
        emb = self.embedding(x)                        # [B, T, E]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        emb = torch.cat([cls_tokens, emb], dim=1)      # prepend CLS
        seq_len = emb.size(1)
        emb = emb + self.pos_encoding[:, :seq_len, :]
        out = emb
        for block in self.transformer_blocks:
            out = block(out)
        cls_out = out[:, 0, :]   # take CLS only
        return self.fc(cls_out)

# ------------------------------
# Load model
# ------------------------------
model = TransformerClassifier(
    vocab_size=len(vocab) + 2,
    embed_dim=64,
    num_heads=4,
    hidden_dim=128,
    num_classes=5,
    num_layers=2,
    max_len=max_len
)

model.load_state_dict(torch.load("sentiment_model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------------
# Prediction functions
# ------------------------------
def predict_review_chunked(review, max_len=100):
    tokens = simple_tokenize(review)
    if not tokens:
        return "No valid tokens found"

    chunks = [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]
    preds, confs = [], []

    with torch.no_grad():
        for chunk in chunks:
            encoded = encode_tokens(chunk)
            input_tensor = torch.tensor([encoded], dtype=torch.long)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            preds.append(pred.item() + 1)
            confs.append(conf.item())

    best_idx = np.argmax(confs)
    return f"{preds[best_idx]} stars (confidence: {confs[best_idx]:.2f})"

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Review Sentiment Rating")
st.write("Enter a product review and get a rating from 1 ⭐ (bad) to 5 ⭐ (good).")

user_input = st.text_area("Type your review here:", height=150)

if st.button("Predict"):
    if user_input.strip() != "":
        rating = predict_review_chunked(user_input)
        st.success(f"Predicted Rating: {rating}")
    else:
        st.warning("⚠️ Please enter some text to predict.")
