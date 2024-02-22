import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.stats import zscore
import pywt

def normr(matrix):
    row_norms = np.linalg.norm(matrix, axis=1)
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix
def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix

REGRESSOR_COUNT = 4

def dynSys(var_dat=None,epoch_dat=None,region_dat=None,sampling_time=.0025):
    # Initialize some example data (Replace these with your actual data)
    # Reading xarray Data from NetCDF file
    #ds = xr.open_dataset('sub-032304_label_ts_EC.nc')
    #time_dat = ds['time'].values
    #var_dat = ds['__xarray_dataarray_variable__'].values
    #epoch_dat = ds['epoch'].values
    #region_dat = ds['region'].values
    var_dat = np.transpose(var_dat, (2, 1, 0))
    #print(var_dat.shape)

    #ts = 0.0025

    Coupling_strengths = []
    condition_numbers = []

    for k in range(len(epoch_dat)):
        
        x = var_dat[:, :, k]

        y = []

        # Inferring Interactions
        #print(x.shape)
        for j in range(len(x[:,0]) - 1):
            y.append((x[j + 1, :] - x[j, :]) / sampling_time)

        # Inferring Interactions
        #y = (x[1:, :] - x[:-1, :])/sampling_time

            
        y = np.array(y)

        # Determine the actual number of decomposition levels for the first signal
        # as an example to estimate levels for others
        sample_coeffs = pywt.wavedec(x[:-1, 0], 'db4', mode='symmetric')
        decomposition_level = len(sample_coeffs)  # Actual number of levels obtained

        # Initialize storage for max length of each decomposition level and feature vectors
        max_lengths = [0] * decomposition_level
        all_features_by_level = [[] for _ in range(decomposition_level)]

        for j in range(x.shape[1]):
            coeffs = pywt.wavedec(x[:-1, j], 'db4', mode='symmetric', level=decomposition_level-1)
            for level, coeff in enumerate(coeffs):
                all_features_by_level[level].append(coeff)
                max_lengths[level] = max(max_lengths[level], coeff.size)

        # Pad coefficients within each level to their max length and stack them
        padded_features = [np.vstack([np.pad(coeff, (0, max_len - coeff.size), 'constant')
                                      for coeff in level_features])
                           for level_features, max_len in zip(all_features_by_level, max_lengths)]

        # Combine padded features from all levels
        phix_list = np.hstack(padded_features)

        # Add an intercept term and prepare for regression
        phix = np.column_stack([np.ones((phix_list.shape[0], 1)), phix_list])

        # Regression analysis
        
        # Fitting
        inverse = np.linalg.pinv(phix)
        W = inverse @ y
        condition_numbers.append(np.linalg.cond(inverse))
        #print(W.shape)


        L = []
        # Initialize G with zeros (or any other value you prefer)
        #G = np.zeros((len(phix_list[1]), W.shape[1] - 1, x.shape[1]))
        for i in range(x.shape[1]):  # Assuming x is a 2D NumPy array
            #f, g = decouple(i, W)  # Assuming decouple is a function defined similarly as above
            g = W[:, i]  # Copying to avoid modifying the original array
            # Remove the first element from g
            g = np.delete(g, 0)
            # Reshape g
            g = np.reshape(g, (REGRESSOR_COUNT, len(g) // REGRESSOR_COUNT))
            #print(g.shape)
            gh_i = np.sqrt(np.sum(g ** 2, axis=0))

            #Change the ith element to a zero
            gh_i[i] = 0

            #print(gh_i.shape)
            #Gh.append(gh_i)
                
            L.append(gh_i)

        # Convert lists to NumPy arrays for further calculations
        #F = np.array(F)
        #G = np.array(G)
        #Gh = np.array(Gh)
        L = np.array(L).T
        Coupling_strengths.append(L)

    Coupling_strengths = np.array(Coupling_strengths).T
    Coupling_strengths = normc(Coupling_strengths.mean(2))

    return Coupling_strengths, np.mean(condition_numbers)