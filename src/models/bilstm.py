import torch
import torch.nn as nn

class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout_rate=0.0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x):
        x = self.embedding(x)
        
        output, (h_n, c_n) = self.bilstm(x)

        forward = h_n[-2]
        backward = h_n[-1]
        h = torch.cat((forward, backward), dim=1)

        x = self.dropout(h)

        x = self.fc(x).squeeze(1)

        return x