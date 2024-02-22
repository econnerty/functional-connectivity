import matplotlib.pyplot as plt

# Data
terms = [4,25,50]
fourier_condition_numbers = [69491575.67322358, 2.903092411421319e+19]
laguerre_condition_numbers = [1715872186.7555075, 1.0444880573743189e+22]
fourier_mse = [88.11495732062713, 83.3266129853063]
laguerre_mse = [84.64736881562837, 82.72226028977408]

# Plotting Condition Numbers
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(terms, fourier_condition_numbers, label='Fourier Basis', marker='o')
plt.plot(terms, laguerre_condition_numbers, label='Laguerre Basis', marker='s')
plt.yscale('log')
plt.xlabel('Number of Terms')
plt.ylabel('Condition Number')
plt.title('Condition Number Comparison')
plt.legend()

# Plotting MSE
plt.subplot(1, 2, 2)
plt.plot(terms, fourier_mse, label='Fourier Basis', marker='o')
plt.plot(terms, laguerre_mse, label='Laguerre Basis', marker='s')
plt.xlabel('Number of Terms')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison')
plt.legend()

plt.tight_layout()
plt.show()