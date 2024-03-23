import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns   
from tqdm.auto import tqdm


# Manual ESN Implementation
class SimpleESN:
    def __init__(self, n_reservoir, spectral_radius, sparsity,rho=0.9, noise=0.1,alpha=1.0,leaky_rate=0.1):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.state = np.zeros(n_reservoir)
        self.W = None
        self.W_in = None
        self.W_out = None
        self.rho = rho
        self.noise = noise
        self.alpha = alpha
        self.leaky_rate = leaky_rate

    def initialize_weights(self):
        # Internal weights
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Set sparsity
        W[np.random.rand(*W.shape) > self.sparsity] = 0
        # Scale weights to have the desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        # Input weights
        self.W_in = np.random.rand(self.n_reservoir, 1) * 2 - 1

    def update_state(self, input):
        pre_activation = np.dot(self.W, self.state) + np.dot(self.W_in, input)
        updated_state = np.tanh(pre_activation)
        self.state = self.leaky_rate * updated_state + (1 - self.leaky_rate) * self.state


    def train(self, inputs, outputs):
         # Initialize weights
        self.initialize_weights()
        
        # Collect states for training
        states = []
        for input in inputs:
            self.update_state(input)
            states.append(self.state.copy())
            """if np.isnan(self.state).any():
                print("nan in state")
                self.state = np.zeros(self.n_reservoir)"""
        
        # Train the readout layer
        reg = LinearRegression()
        reg.fit(states, outputs)
        self.W_out = reg.coef_
    
    def predict(self, inputs):
        states = []
        for input in inputs:
            self.update_state(input)
            states.append(self.state.copy())
        
        return np.dot(states, self.W_out.T)

# Function to train an ESN and calculate MSE for a given input-output pair
def train_and_evaluate_epoch(input_series, target_series):
    """
    Trains an ESN for a single epoch and computes the MSE.
    """
    esn = SimpleESN(n_reservoir=20, spectral_radius=1.2, sparsity=0.3)
    esn.train(input_series, target_series)
    predictions = esn.predict(input_series).flatten()
    mse = np.mean((predictions - target_series)**2)
    return mse

def compute_adjacency_matrix_for_epoch(epoch_data):
    """
    Computes the adjacency matrix for a single epoch.
    """
    n_series = epoch_data.shape[0]  # Number of time series
    #print(n_series)
    mse_results = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(n_series):
            if i != j:
                # Example check before training
                #print(epoch_data[i, :].shape)
                mse_results[i, j] = train_and_evaluate_epoch(epoch_data[i, :, np.newaxis], epoch_data[j, :])

    # Invert MSE for adjacency matrix (higher value means stronger predictive power)
    adjacency_matrix = np.where(mse_results != 0, 1 / mse_results, 0)
    return adjacency_matrix

def dynSys(var_dat=None,epoch_dat=None,region_dat=None,sampling_time=.004):
    # Main process
    #var_dat = np.transpose(var_dat, (2, 1, 0))
    n_epochs = var_dat.shape[0]
    n_series = var_dat.shape[1]  # Assuming var_dat is now [epochs, time_points, series]
    #print(n_epochs)
    #print(n_series)
    # Initialize a list to hold all adjacency matrices
    all_adjacency_matrices = []
    for k in tqdm(range(n_epochs)):
        epoch_data = var_dat[k, :, :]  # Extract data for the k-th epoch

        adj_matrix = compute_adjacency_matrix_for_epoch(epoch_data)
        all_adjacency_matrices.append(adj_matrix)

    # Average the adjacency matrices across all epochs
    avg_adjacency_matrix = np.mean(np.array(all_adjacency_matrices), axis=0)
    #Min max normalize the matrix
    avg_adjacency_matrix = (avg_adjacency_matrix - np.min(avg_adjacency_matrix)) / (np.max(avg_adjacency_matrix) - np.min(avg_adjacency_matrix))
    output = sns.heatmap(avg_adjacency_matrix, xticklabels=region_dat, yticklabels=region_dat)
    output.get_figure().savefig(f'./dynsys/reservoir.png')
