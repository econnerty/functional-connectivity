#Author: Erik Connerty
#Date: 3/23/2024

#Pairwise Reservoir Approximation

import numpy as np
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso
import matplotlib.pyplot as plt
import seaborn as sns   
from tqdm.auto import tqdm

def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix
def normr(matrix):
    row_norms = np.linalg.norm(matrix, axis=1)
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix

# Manual ESN Implementation
class SimpleESN:
    def __init__(self, n_reservoir, spectral_radius, sparsity,rho=0.9, noise=0.1,alpha=1.0,leaky_rate=0.5,lag = -2):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.state = np.zeros(n_reservoir)
        self.W = None
        self.W_in = None
        self.W_bias = None  # Bias weights
        self.W_out = None
        self.rho = rho
        self.noise = noise
        self.alpha = alpha
        self.leaky_rate = leaky_rate
        self.lag = lag
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self):
        # Internal weights
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[np.random.rand(*W.shape) < self.sparsity] = 0  # Set sparsity
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)  # Scale weights
        
        # Input weights
        self.W_in = np.random.rand(self.n_reservoir, 1) * 2 - 1
        
        # Initialize bias weights
        self.W_bias = np.random.rand(self.n_reservoir) * 2 - 1

    def update_state(self, input):
        # Include bias term in the pre-activation signal
        pre_activation = (1-self.leaky_rate) * np.dot(self.W, self.state) + self.leaky_rate * (np.dot(self.W_in, input) + self.W_bias)
        self.state = np.tanh(pre_activation)
        #self.state = self.leaky_rate * updated_state + (1-self.leaky_rate) * self.state

    
    def predict(self, inputs):
        return np.dot(inputs, self.W_out.T)

def train_and_evaluate_with_states(esn, input_states, target_series):
    """
    Trains an ESN with precomputed input and target states, then computes a modified MSE.
    De-means predictions and target before computing MSE to focus on pattern similarity.
    """
    input_states = np.array(input_states)

    # Train the readout layer with ElasticNet
    regressor = Ridge(alpha=2)
    regressor.fit(input_states, target_series)
    esn.W_out = regressor.coef_

    # Predict using the biased target states
    predictions = esn.predict(input_states).flatten()

    # Compute MSE on de-meaned series
    mse = np.mean((predictions - target_series) ** 2)

    # Get signal to noise ratio
    mse = mse + 1e-20  # Avoid division by zero
    snr = np.mean(predictions ** 2) / mse
    return snr



def compute_adjacency_matrix_for_epoch(epoch_data, lag=0,sampling_time=0.01,num_reservoir=25):
    """
    Computes the adjacency matrix for a single epoch, optimizing by calculating
    reservoir states only once for each 'i' in the outer loop, and reusing these states
    for all 'j' comparisons.
    """
    n_series = epoch_data.shape[0]  # Number of time series
    mse_results = np.zeros((n_series, n_series))

    sparsity = .3
    if num_reservoir <= 10:
        sparsity = 0.0

    # Initialize the ESN instance
    esn = SimpleESN(n_reservoir=num_reservoir, spectral_radius=1.0, sparsity=sparsity,leaky_rate=0.8)
    esn.initialize_weights()  # Initialize weights once at the start

    for i in range(n_series):
        esn.state = np.zeros(esn.n_reservoir)  # Reset state only here, at the start of each 'i' loop

        # Collect states for the 'i' series
        input_states_i = []
        for input_val in epoch_data[i, :, np.newaxis]:
            esn.update_state(input_val)
            input_states_i.append(esn.state.copy())

        for j in range(n_series):
            target_series = epoch_data[j, :]

            #Calculate the time derivative of the target series
            target_series = np.diff(target_series)/sampling_time
            
            mse_results[j, i] = train_and_evaluate_with_states(esn, input_states_i[:-1], target_series)

    return mse_results


#Pairwise Reservoir Approximation
def PRA(var_dat=None,sampling_frequency=100,num_reservoir=15):
    #Calculate the sampling time
    sampling_time = 1/sampling_frequency
    # Main process
    n_epochs = var_dat.shape[0]
    n_series = var_dat.shape[1]  # Assuming var_dat is now [epochs, time_points, series]
    # Initialize a list to hold all adjacency matrices
    all_adjacency_matrices = []
    for k in tqdm(range(n_epochs)):
        epoch_data = var_dat[k, :, :]  # Extract data for the k-th epoch

        adj_matrix = compute_adjacency_matrix_for_epoch(epoch_data,sampling_time=sampling_time,num_reservoir=num_reservoir)
        all_adjacency_matrices.append(adj_matrix)

    # Average the adjacency matrices across all epochs
    avg_adjacency_matrix = np.mean(np.array(all_adjacency_matrices), axis=0)

    # Set diagonal to zero
    np.fill_diagonal(avg_adjacency_matrix, 0)
    #Min max normalize the matrix
    avg_adjacency_matrix = (avg_adjacency_matrix - np.min(avg_adjacency_matrix)) / (np.max(avg_adjacency_matrix) - np.min(avg_adjacency_matrix))

    return avg_adjacency_matrix

