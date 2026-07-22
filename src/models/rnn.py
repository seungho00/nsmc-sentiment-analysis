import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout_rate=0.0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x):
        x = self.embedding(x)
        
        output, h_n = self.rnn(x)

        x = self.dropout(h_n[-1])

        x = self.fc(x).squeeze(1)

        return x