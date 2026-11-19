
import torch
import torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.linear = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, x):
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        tok_embeddings = self.tok_emb(x)  # (batch_size, seq_length, emb_dim)

        # NOTE: Understand what is happening with positional embeddings here.
        pos_indices = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_emb(pos_indices)  # (batch_size, seq_length, emb_dim)
        ## 

        embeddings = tok_embeddings + pos_embeddings  # (batch_size, seq_length, emb_dim)
        logits = self.linear(embeddings)  # (batch_size, seq_length, vocab_size)
        return logits
