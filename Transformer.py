import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

# Assuming each time series is of length T=10 for simplicity and 3 time series
T = 1000
num_series = 3
batch_size = 1

# Step 1: Synthetic Data with simple masking
data = torch.randn(batch_size, T, num_series)  # Adjusted shape for embedding compatibility
mask = torch.rand(batch_size, T, num_series) < 0.8  # 80% chance of being True
masked_data = data.clone()
masked_data[~mask] = 0  # Apply masking


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self._last_attn_weights = None  # Store last attention weights here

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        self._last_attn_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    @property
    def last_attn_weights(self):
        return self._last_attn_weights
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src = layer(src)
        output = self.decoder(src)
        return output

    def get_last_attn_weights(self):
        # Assuming we want the attention weights from the last layer for simplicity
        return self.transformer_encoder[-1].last_attn_weights

model = TransformerModel(input_dim=num_series, d_model=512, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 2: Training with masked data
for _ in tqdm(range(200)):  # Small number of epochs for demonstration
    model.train()
    optimizer.zero_grad()
    output = model(masked_data)
    loss = loss_function(output[mask], data[mask])  # Compute loss only on masked elements
    loss.backward()
    optimizer.step()
    #print(model.get_last_attn_weights().shape)
    print(f"Loss: {loss.item()}")

print("Training completed")
attn_weights = model.get_last_attn_weights() # Get attention weights
# Example: Extract and interpret attention weights
# attn_weights shape: (batch_size, nhead, seq_len, seq_len)
print("Attention weights shape:", attn_weights.shape)
# For simplification, we are using data from the first head
attention_matrix = attn_weights[0, 0].detach().numpy()
print("Attention matrix for the first head:", attention_matrix)
