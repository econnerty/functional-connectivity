import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.stats import zscore
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spilu
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spilu

def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix
def calculate_mse(y_actual, y_pred):
    residuals = y_actual - y_pred
    mse = np.mean(np.square(residuals))
    return mse
def calculate_snr(y_actual, y_pred):
    residuals = y_actual - y_pred
    mse = np.mean(np.square(residuals))
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
    plt.savefig('./dynsys/approximated_integral_10.png')
REGRESSOR_COUNT = 10
def dynSys(var_dat=None,epoch_dat=None,region_dat=None,sampling_time=.004):
    # Initialize some example data (Replace these with your actual data)
    # Reading xarray Data from NetCDF file
    var_dat = np.transpose(var_dat, (2, 1, 0))

    Coupling_strengths = []
    condition_numbers = []
    mse = []
    snr = []

    for k in tqdm(range(len(epoch_dat))):
        
        x = var_dat[:, :, k]


        y = []
        # Inferring Interactions
        for j in range(len(x[:,0]) - 1):
            y.append((x[j + 1, :] - x[j, :]) / sampling_time)
            
        y = np.array(y)
            
        # Regressor generation
        phix_list = []
        for j in range(x.shape[1]):
            for n in range(1, REGRESSOR_COUNT + 1):
                if n % 2 == 1:  # For odd indices, use sine
                    term = np.sin((n // 2 + 1) * x[:-1, j])
                else:  # For even indices, use cosine
                    term = np.cos((n // 2) * x[:-1, j])
                phix_list.append(term)
        #print(len(phix_list[0]))
        phix_array = np.array(phix_list).T
        #print(phix_array.shape)

        phix = np.column_stack([np.ones((len(x) - 1, 1)), phix_array])

        #SVD Preconditioning
        U, s, Vt = np.linalg.svd(phix, full_matrices=False)
        # Regularize small singular values
        threshold = 10.0
        s_reg = np.array([max(x, threshold) for x in s])  # Regularize singular values
        S_reg = np.diag(s_reg)  # Construct a diagonal matrix with the regularized singular values
        phix = np.dot(U, np.dot(S_reg, Vt))  # Reconstruct the modified matrix

        # Fitting
        inverse = np.linalg.pinv(phix)
        W = inverse @ y
        y_pred = phix @ W
        mse.append(calculate_mse(y, y_pred))
        condition_numbers.append(np.linalg.cond(phix))
        snr.append(calculate_snr(y, y_pred))

        #Plot predictions vs actual
        #print(y_pred.shape)
        """plt.plot(y[:,0],label='Actual')
        plt.plot(y_pred[:,0],label='Predicted')
        plt.legend()
        plt.savefig(f'./dynsys/{k}_dynsys_4.png')
        plt.close()"""
        #integrate_and_plot(y_pred, x, sampling_time)

        #Plot the FFT of the predicted and actual
        """plt.plot(np.abs(np.fft.fft(y[:,0])),label='Actual')
        plt.plot(np.abs(np.fft.fft(y_pred[:,0])),label='Predicted')
        plt.legend()
        plt.savefig(f'./dynsys/{k}_dynsys_fft.png')
        plt.close()"""

        L = []
        # Initialize G with zeros (or any other value you prefer)
        for i in range(x.shape[1]):  # Assuming x is a 2D NumPy array
            g = W[:, i]  # Copying to avoid modifying the original array
            # Remove the first element from g
            g = np.delete(g, 0)
            # Reshape g
            g = np.reshape(g, (REGRESSOR_COUNT, len(g) // REGRESSOR_COUNT))
            gh_i = np.sqrt(np.sum(g ** 2, axis=0))
            #print(gh_i)
            #Change the ith element to a zero
            
            #gh_i[i] = 0
                
            L.append(gh_i)

        # Convert lists to NumPy arrays for further calculations
        L = np.array(L)
        Coupling_strengths.append(L)

    Coupling_strengths = np.array(Coupling_strengths)
    #print(Coupling_strengths.shape)
    Coupling_strengths = normc(Coupling_strengths.mean(0))
    #Min max normalize the array
    Coupling_strengths = (Coupling_strengths - Coupling_strengths.min()) / (Coupling_strengths.max() - Coupling_strengths.min())
    #Zero out the diagonal
    #print(Coupling_strengths)
    np.fill_diagonal(Coupling_strengths, 0)



    return Coupling_strengths, np.mean(condition_numbers), np.mean(mse),np.mean(snr)