import math
import numpy as np
from matplotlib import pyplot as plt

def movingAverage(signal: np.array, h: int) -> np.array:
    """
        Simple moving average filter.
        
        Moving average filter made for smoothing signal's noise.
        !Important!
        This function cuts `h` first samples of the `signal`. For 
        input `signal` of size `(1, 628)` output signal will have 
        size `(1, 628-h)` and first `h` samples will be lost.
        
        Arguments:
        signal: np.array - signal to filter
        h: int - number of samples to consider
        
        Returns:
        np.array - filtered signal of size `(n-h)` 
    """
    
    filtered_signal = np.zeros(signal.size - h)
    for i in range(filtered_signal.size):
        filtered_signal[i] = 1/h * np.sum(signal[i:h+i])
    return filtered_signal

def MSE(first: np.array, second: np.array) -> float:
    """"
        Calculates the Mean Squared Error of two passed arrays.
        
        Arguments:
        first: np.array - first array
        second: np.array - second array
        
        Returns:
        float - MSE of two passed arrays
    """
    if first.size != second.size:
        raise ValueError(f"Number of samples in first and second \
            arrays must match! (first={(first.size)}, second={(second.size)})")
    return np.mean(np.square(first-second))
    

"""
    Main function variables:
    
    h_array: np.array - array of h values ['h_min', 'h_max') with step 1
    t: np.array - array of original function argument values
    function: np.array - array of original function values
"""
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
    noise_scale: float (0, +inf] - value for scailing noise applied
                                   to each sample
    noised_samples: np.array - array of sample values with noise
"""
n_samples = 250
variance_min, variance_max = 0, 2.1
variance = np.arange(variance_min, variance_max, 0.1)
samples_t = np.linspace(0, 2*np.pi, n_samples)
samples_array = np.sin(samples_t)



h_min = 1
h_max = 31
h_array = np.arange(h_min, h_max, 1) 
mse_array = np.zeros(h_array.size) 


h_for_variance = np.zeros(variance.size)

for i, v in enumerate(variance):
    noise =  np.sqrt(v) * np.random.randn(n_samples) 
    noised_samples = samples_array + noise
    
    for h in h_array:
        filtered = movingAverage(noised_samples, h)
        mse_array[h-1] = MSE(filtered, samples_array[h:])
    
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
