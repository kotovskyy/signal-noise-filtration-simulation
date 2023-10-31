import numpy as np
from matplotlib import pyplot as plt

import utils
    
    
"""
    Main function variables:
    
    h_array: np.array - array of h values ['h_min', 'h_max') with step 1
    t: np.array - array of original function argument values
    function: np.array - array of original function values
"""
h_min = 1
h_max = 31
h_array = np.arange(h_min, h_max, 1) 


t = np.linspace(0, 2*np.pi, 1000)
function = np.sin(t)

"""
    Sampling variables:
    
    n_samples: int - number of samples. 
               Must be such that: 1000 % n_samples = 0;
    samples_t: np.array - array of time value for each sample
    samples_array: np.array - array of values of original function 
                              for each samples_t value
    noise: np.array - array of random noise values from normal distribution
    std_dev: float (0, +inf] - standard deviation of generated noise
    noised_samples: np.array - array of sample values with noise
"""

n_samples = 250
samples_t = np.linspace(0, 2*np.pi, n_samples)
samples_array = np.sin(samples_t)
std_dev = 0.5
noise = std_dev * np.random.randn(n_samples) 
noised_samples = samples_array + noise

# MSE values for each 'h' in 'h_array'
mse_array = np.zeros(h_array.size) 

for h in h_array:
    filtered = utils.movingAverage(noised_samples, h)
    mse_array[h-1] = utils.MSE(filtered, samples_array[h:])


min_mse = np.argmin(mse_array)
h = h_array[min_mse]
mse = mse_array[min_mse]
print(f"MIN MSE = {mse}, h = {h}")

filtered = utils.movingAverage(noised_samples, h)


plt.figure(figsize=(8, 4))
plt.plot(t, function,
         label="Funkcja oryginalna")
plt.plot(samples_t[h:], filtered, '--g',
         label=f"Pomiary uśrednione, h={h}")
plt.scatter(samples_t, noised_samples, 
            color="red", 
            marker='.',
            label=f"Pomiary zaszumione, Var(Z)={std_dev**2:.3}")
plt.yticks(np.arange(-1.25, 1.25, 0.25))
plt.xticks(np.arange(0.0, 6.5, 0.5))
plt.xlabel("t")
plt.ylabel("f(t)")
plt.grid()
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(h_array, mse_array, '--o')
plt.xlabel("h")
plt.ylabel("MSE(h)")
plt.xticks(np.arange(0, h_max, 2))
plt.grid()
plt.show()
