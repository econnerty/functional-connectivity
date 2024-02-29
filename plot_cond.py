import matplotlib.pyplot as plt

# Updated data based on the new values provided
terms = [4, 25, 50]
fourier_condition_numbers = [78648487.35031119, 3.205764418155434e+19, 6.958307848897716e+19]
laguerre_condition_numbers = [1562651533.482242, 6.0976191954410185e+19, 3.0147981223417164e+22]
fourier_mse = [86.896356265107, 82.268481768314, 76.01416847793362]
laguerre_mse = [85.31506198556278, 83.05880035786583, 90.20651854426366]


# Plotting the updated data
plt.figure(figsize=(14, 6))

# Condition Numbers Comparison
plt.subplot(1, 2, 1)
plt.plot(terms, fourier_condition_numbers, label='Fourier Basis', marker='o', color='navy')
plt.plot(terms, laguerre_condition_numbers, label='Laguerre Basis', marker='s', color='crimson')
plt.yscale('log')
plt.xlabel('Number of Terms')
plt.ylabel('Condition Number')
plt.title('Condition Number Comparison')
plt.legend()

# MSE Comparison
plt.subplot(1, 2, 2)
plt.plot(terms, fourier_mse, label='Fourier Basis', marker='o', color='navy')
plt.plot(terms, laguerre_mse, label='Laguerre Basis', marker='s', color='crimson')
plt.xlabel('Number of Terms')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison')
plt.legend()

plt.tight_layout()
plt.show()