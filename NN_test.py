import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm

class TimeSeriesToGraphModel(nn.Module):
    def __init__(self, num_nodes, hidden_size=128, seq_len=4000, num_layers=2):
        super(TimeSeriesToGraphModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        # Feature extraction RNN
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Adjust the size for adj_predictor based on your strategy for node pair features
        self.adj_predictor = nn.Sequential(nn.Linear(hidden_size, num_nodes), nn.Sigmoid())
        # Decoder RNN
        self.decoder_rnn = nn.GRU(input_size=hidden_size + num_nodes, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, hn = self.rnn(x)
        hn = hn[-1]  # Last layer's output, shape: [batch_size, hidden_size]
        
        # Simplified adjacency matrix prediction assuming independence of nodes
        # Consider a more complex interaction model for real applications
        adj_matrix = self.adj_predictor(hn).unsqueeze(1)  # Assuming independence, shape: [batch_size, 1, num_nodes]
        
        # Repeat adjacency info across seq_len for concatenation
        adj_matrix_repeated = adj_matrix.repeat(1, seq_len, 1)  # Shape: [batch_size, seq_len, num_nodes]
        
        # Preparing combined input for the decoder
        rnn_output, _ = self.rnn(x)  # Obtain RNN output for use in decoding
        combined_input = torch.cat((rnn_output, adj_matrix_repeated), dim=2)  # Concat features and adjacency
        
        decoded, _ = self.decoder_rnn(combined_input)  # Shape: [batch_size, seq_len, hidden_size]
        decoded = self.decoder_fc(decoded)  # Shape: [batch_size, seq_len, 1]
        
        return decoded, adj_matrix.squeeze(1)  # Squeeze to remove the singleton dimension


# Training and adjacency matrix extraction
def dynSys(var_dat=None,epoch_dat=None,region_dat=None,sampling_time=.004):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = TimeSeriesToGraphModel(num_nodes=var_dat.shape[1], hidden_size=128, seq_len=var_dat.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(1):
        epoch_loss = 0
        for k in tqdm(range(var_dat.shape[2]), desc=f'Epoch {epoch+1}/{1}'):  # Iterate over the "epochs" dimension
            x = torch.tensor(var_dat[:, :, k], dtype=torch.float32).unsqueeze(-1).to(device)  # Shape adjustment
            optimizer.zero_grad()
            output, adj_matrix = model(x)
            loss = loss_function(output, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / var_dat.shape[2]}")
    
    # Extract the learned adjacency matrix after training
    collected_adj_matrices = []  # List to store adjacency matrices
    model.eval()
    for k in tqdm(range(var_dat.shape[2])):  # Iterate over the "epochs" dimension
        x = torch.tensor(var_dat[:, :, k], dtype=torch.float32).unsqueeze(-1).to(device)  # Shape adjustment
        _, adj_matrix = model(x)
        collected_adj_matrices.append(adj_matrix.detach().cpu().numpy())
    
    # Average the collected adjacency matrices
    average_adj_matrix = np.mean(np.array(collected_adj_matrices), axis=0)
    print(average_adj_matrix)
    return average_adj_matrix

