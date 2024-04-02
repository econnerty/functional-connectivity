import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.stats import zscore
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spilu
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spilu
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso

def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix

def normr(matrix):
    row_norms = np.linalg.norm(matrix, axis=1)
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix
def calculate_mse(y_actual, y_pred):
    residuals = y_actual - y_pred
    mse = np.mean(np.square(residuals))
    return mse
def calculate_snr(y_actual, y_pred):
    residuals = y_actual - y_pred
    mse = np.mean(np.square(residuals))
    mse += 1e-5  # Add a small value to avoid division by zero
    snr = 10 * np.log10(np.mean(np.square(y_pred)) / mse)
    return snr

def integrate_and_plot(y_pred, x, sampling_time):
    """
    Integrate the predicted derivative values and plot against the original time series data.
    
    Args:
        y_pred: The predicted derivatives of the state variables with shape [time_points-1, channels].
        x: Original time series data with shape [time_points, channels].
        sampling_time: Time step between observations.
    """
    time_points, channels = x.shape
    x_solved = np.zeros_like(x)
    
    # For each channel, integrate y_pred to get x_solved
    for channel in range(channels):
        # Initialize the first value of x_solved for each channel as the initial state from x
        x_solved[0, channel] = x[0, channel]
        
        # Integrate using the forward Euler method
        for t in range(1, time_points):
            x_solved[t, channel] = x_solved[t-1, channel] + y_pred[t-1, channel] * sampling_time

    # Plotting
    fig, axs = plt.subplots(10, 7, figsize=(20, 28))  # Adjust subplot grid as needed
    for i, ax in enumerate(axs.flatten()):
        if i < channels:
            ax.plot(x[:, i], label='Original', color='blue')
            ax.plot(x_solved[:, i], label='Predicted', linestyle='--', color='red')
            ax.set_title(f'Channel {i+1}')
            ax.legend()
        else:
            ax.axis('off')
    plt.title('Approximated Integral of the Time Series Data')
    plt.tight_layout()
    plt.show()


# Manual ESN Implementation
class SimpleESN:
    def __init__(self, n_reservoir, spectral_radius, sparsity,rho=0.9, noise=0.1,alpha=1.0,leaky_rate=0.1,lag = -2):
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
        self.lag = lag

    def initialize_weights(self):
        # Internal weights
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Set sparsity
        W[np.random.rand(*W.shape) < self.sparsity] = 0
        # Scale weights to have the desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        # Input weights
        self.W_in = np.random.rand(self.n_reservoir, 1) * 2 - 1


    def update_state(self, input):
        pre_activation = np.dot(self.W, self.state) + np.dot(self.W_in, input)
        updated_state = np.tanh(pre_activation)
        self.state = (1 - self.leaky_rate) * updated_state +  self.leaky_rate * self.state


    def train(self, inputs):
         # Initialize weights
        #self.initialize_weights()
        
        # Collect states for training
        states = []
        for input in inputs:
            self.update_state(input)
            states.append(self.state.copy())
    
    def predict(self, inputs):
        return np.dot(inputs, self.W_out.T)


NUMBER_OF_NODES = 1
def dynSys(var_dat=None,epoch_dat=None,region_dat=None,sampling_time=.004):
    # Initialize some example data (Replace these with your actual data)
    # Reading xarray Data from NetCDF file
    var_dat = np.transpose(var_dat, (2, 1, 0))

    Coupling_strengths = []
    condition_numbers = []
    mse = []
    snr = []


    for k in tqdm(range(var_dat.shape[2])):
        
        x = var_dat[:, :, k]

        y = []
        # Inferring Interactions
        for j in range(x.shape[1]):
            y.append(np.diff(x[:, j]) / sampling_time)
            #y.append((x[j + 1, :] - x[j, :]) / sampling_time)
            
        y = np.array(y).T
        #y = x

        # Initialize a list to store reservoir states for each time series
        all_states = []
        
        for j in range(x.shape[1]):  # Iterate over each time series
            esn = SimpleESN(n_reservoir=NUMBER_OF_NODES, spectral_radius=1.0, sparsity=0.0,leaky_rate=0.5)
            esn.initialize_weights()

            single_series_data = x[:-1, j]
            #single_series_data = x[:, j]

            # Collect states for the current time series
            states = []
            for input_val in single_series_data:
                esn.update_state([input_val])  # Ensure input_val is in the expected shape, e.g., [input_val] or [[input_val]]
                states.append(esn.state.copy())

            all_states.append(np.array(states))

        # Reshape and concatenate states
        # Each item in all_states is of shape [time_points, n_reservoir], so stack along the second dimension
        phix = np.hstack(all_states)  # This forms a matrix of shape [time_points, n_reservoir*num_series]
        
        # Add a column of ones to the regressor matrix
        phix = np.insert(phix, 0, 1, axis=1)

        regressor = Ridge(alpha=10)
        #W = inverse @ y
        W = regressor.fit(phix, y).coef_.T
        y_pred = phix @ W
        condition_number = np.linalg.cond(phix)
        mse.append(calculate_mse(y, y_pred))
        condition_numbers.append(condition_number)
        snr.append(calculate_snr(y, y_pred))
        """plt.plot(y[:,0],label='Actual')
        plt.plot(y_pred[:,0],label='Predicted')
        plt.legend()
        plt.show()
        #plt.savefig(f'./dynsys/{k}_dynsys_rc_{NUMBER_OF_NODES}.png')
        plt.close()"""
        #integrate_and_plot(y_pred, x, sampling_time)

        L = []
        # Initialize G with zeros (or any other value you prefer)
        for i in range(x.shape[1]):  # Assuming x is a 2D NumPy array
            g = W[:, i]  # Copying to avoid modifying the original array
            # Remove the first element from g
            g = np.delete(g, 0)
            # Reshape g
            g = np.reshape(g, (NUMBER_OF_NODES, len(g) // NUMBER_OF_NODES))
            gh_i = np.sqrt(np.sum(g ** 2, axis=0))
            #Change the ith element to a zero
            gh_i[i] = 0
                
            L.append(gh_i)

        # Convert lists to NumPy arrays for further calculations
        L = np.array(L)
        Coupling_strengths.append(L)

    Coupling_strengths = np.array(Coupling_strengths).mean(0)
    #print(Coupling_strengths.shape)
    Coupling_strengths = normc(Coupling_strengths)
    #Min max normalize the array
    Coupling_strengths = (Coupling_strengths - Coupling_strengths.min()) / (Coupling_strengths.max() - Coupling_strengths.min())
#


    return Coupling_strengths, np.mean(condition_numbers), np.mean(mse),np.mean(snr)