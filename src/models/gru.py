import torch
import torch.nn as nn

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x):
        x = self.embedding(x)
        
        output, h_n = self.gru(x)

        x = self.fc(h_n[-1]).squeeze(1)

        return x