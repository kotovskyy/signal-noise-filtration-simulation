import numpy as np
from matplotlib import pyplot as plt

import utils

"""Creation of input signal"""
t = np.linspace(0, 2*np.pi, 1000)
function = np.sin(t)

"""Create samples of input signal"""
n_samples = 250
variance_min, variance_max = 0, 2.1 # array of variance values
variance = np.arange(variance_min, variance_max, 0.1)
samples_t = np.linspace(0, 2*np.pi, n_samples)
samples_array = np.sin(samples_t)

h_min = 1
h_max = 31
h_array = np.arange(h_min, h_max, 1) # array of 'h' values used in movingAverage 
mse_array = np.zeros(h_array.size) # list of MSE values for different 'h'

h_for_variance = np.zeros(variance.size) # best 'h' for specific variance

for i, v in enumerate(variance):
    noise =  np.sqrt(v) * np.random.randn(n_samples) 
    noised_samples = samples_array + noise
    
    for h in h_array:
        filtered = utils.movingAverage(noised_samples, h)
        mse_array[h-1] = utils.MSE(filtered, samples_array[h:])
    
    min_mse = np.argmin(mse_array)
    h_opt = h_array[min_mse]
    mse = mse_array[min_mse]
    h_for_variance[i] = h_opt
    

plt.figure(figsize=(8, 4))
plt.plot(variance, h_for_variance, '--o')
plt.xlabel("Var(Z)")
plt.ylabel("h")
plt.xticks(np.arange(0, variance_max, 0.1))
plt.grid()
plt.show()
