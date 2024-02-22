import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.stats import zscore
from tqdm.auto import tqdm

def normc(matrix):
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / column_norms[np.newaxis, :]
    return normalized_matrix
def calculate_mse(y_actual, y_pred):
    residuals = y_actual - y_pred
    mse = np.mean(np.square(residuals))
    return mse

REGRESSOR_COUNT = 50
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
    mse = []

    for k in tqdm(range(len(epoch_dat))):
        
        x = var_dat[:, :, k]

        y = []

        # Inferring Interactions
        #print(x.shape)
        for j in range(len(x[:,0]) - 1):
            y.append((x[j + 1, :] - x[j, :]) / sampling_time)

        # Inferring Interactions
        #y = (x[1:, :] - x[:-1, :])/sampling_time

            
        y = np.array(y)


        # Inferring Interactions
        #y = (x[1:, :] - x[:-1, :])/sampling_time

            
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

        """sin_terms = np.sin(x[:-1, :])
        cos_terms = np.cos(x[:-1, :])
        sin2_terms = np.sin(2 * x[:-1, :])
        cos2_terms = np.cos(2 * x[:-1, :])"""
        
        phix_array = np.array(phix_list).T
        #phix_array = np.array([sin_terms, cos_terms, sin2_terms, cos2_terms])

        phix = np.column_stack([np.ones((len(x) - 1, 1)), phix_array])

        # Fitting
        inverse = np.linalg.pinv(phix)
        W = inverse @ y
        y_pred = phix @ W
        mse.append(calculate_mse(y, y_pred))
        condition_numbers.append(np.linalg.cond(phix))
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

    return Coupling_strengths, np.mean(condition_numbers), np.mean(mse)