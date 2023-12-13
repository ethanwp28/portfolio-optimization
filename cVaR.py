# Converting CVaR MATLAB Code to Python:
import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming PL_T and R_T are already defined)
# In Python, you would typically have this data in variables already, or you can load it from a file

epsilon = np.array([0.01, 0.05])

# CVaR of the portfolio Profit and Loss
x = np.sort(PL_T)
Nepsilon = np.ceil(epsilon * len(PL_T)).astype(int)
VaR_PL = -x[Nepsilon - 1]  # Adjusted for Python's 0-based indexing
CVaR_PL = np.array([-np.mean(x[:Nepsilon[0]]), -np.mean(x[:Nepsilon[1]])])

# Plotting CVaR(PL_T)
plt.figure(figsize=(10, 8))
plt.hist(-PL_T, bins=50, density=True)
plt.axvline(x=VaR_PL[0], color='k', linewidth=2)
plt.axvline(x=VaR_PL[1], color='k', linewidth=2, linestyle='--')
plt.axvline(x=CVaR_PL[0], color='r', linewidth=3)
plt.axvline(x=CVaR_PL[1], color='r', linewidth=3, linestyle='--')
plt.legend(['VaR ε = 1%', 'VaR ε = 5%', 'CVaR ε = 1%', 'CVaR ε = 5%'])
plt.title('CVaR of PL_T')
plt.show()

# CVaR of the portfolio returns
x = np.sort(R_T)
VaR_R = -x[Nepsilon - 1]  # Adjusted for Python's 0-based indexing
CVaR_R = np.array([-np.mean(x[:Nepsilon[0]]), -np.mean(x[:Nepsilon[1]])])

# Plotting CVaR(R_T)
plt.figure(figsize=(10, 8))
plt.hist(-R_T, bins=30, density=True)
plt.axvline(x=VaR_R[0], color='k', linewidth=2)
plt.axvline(x=VaR_R[1], color='k', linewidth=2, linestyle='--')
plt.axvline(x=CVaR_R[0], color='r', linewidth=3)
plt.axvline(x=CVaR_R[1], color='r', linewidth=3, linestyle='--')
plt.legend(['VaR ε = 1%', 'VaR ε = 5%', 'CVaR ε = 1%', 'CVaR ε = 5%'])
plt.title('CVaR of R_T')
plt.show()
