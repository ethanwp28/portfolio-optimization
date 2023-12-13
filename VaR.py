# Converting VaR MATLAB Code to Python:
import numpy as np
import matplotlib.pyplot as plt

# Define the number of simulations, current prices, and portfolio allocation
m = 1000
P_0 = np.array([7, 9, 24, 19, 15])
q = np.array([20, 20, 10, 10, 10])

# Generate returns from a multivariate normal distribution
R = np.random.randn(m, len(q))

# Confidence levels
epsilon = np.array([0.01, 0.05])

# Calculate current portfolio value
V_0 = np.dot(P_0, q)

# Calculate matrix of future log-normal prices
P = P_0 * np.exp(R)

# Calculate future portfolio values
V_T = np.dot(P, q)

# Profit & Loss vector
PL_T = V_T - V_0

# Portfolio return vector
R_T = np.log(V_T / V_0)

# VaR calculation for PL_T
x = np.sort(PL_T)
Nepsilon = np.ceil(epsilon * m).astype(int)
Quantile = x[Nepsilon - 1]  # Adjusted for Python's 0-based indexing
VaR_PL = -Quantile

# Plotting VaR(PL_T)
plt.figure(figsize=(10, 8))
plt.hist(-PL_T, bins=50, density=True)
plt.axvline(x=VaR_PL[0], color='k', linewidth=3)
plt.axvline(x=VaR_PL[1], color='k', linewidth=3, linestyle='--')
plt.legend([f'ε = {ep*100:.0f}%' for ep in epsilon])
plt.title('VaR of PL_T')
plt.show()

# VaR calculation for R_T
x = np.sort(R_T)
VaR_R = -x[Nepsilon - 1]  # Adjusted for Python's 0-based indexing

# Plotting VaR(R_T)
plt.figure(figsize=(10, 8))
plt.hist(-R_T, bins=30, density=True)
plt.axvline(x=VaR_R[0], color='k', linewidth=3)
plt.axvline(x=VaR_R[1], color='k', linewidth=3, linestyle='--')
plt.legend([f'ε = {ep*100:.0f}%' for ep in epsilon])
plt.title('VaR of R_T')
plt.show()
