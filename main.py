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
    
    h_array: np.array - array of 'h' values to loop through
    t: np.array - array of original function argument values
    function: np.array - array of original function values
"""
h_min = 1
h_max = 21
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
    noise_scale: float (0, +inf] - value for scailing noise applied
                                   to each sample
    noised_samples: np.array - array of sample values with noise
"""
noise_scale = 0.15
n_samples = 250
samples_t = np.linspace(0, 2*np.pi, n_samples)
samples_array = np.sin(samples_t)
noise = noise_scale * np.random.randn(n_samples) 
noised_samples = samples_array + noise

mse_array = np.zeros(h_array.size) # MSE values for each 'h' in 'h_array'

for h in h_array:
    filtered = movingAverage(noised_samples, h)
    mse_array[h-1] = MSE(filtered, samples_array[h:])


min_mse = np.argmin(mse_array)
h = h_array[min_mse]
mse = mse_array[min_mse]
print(f"MIN MSE = {mse}, h = {h}")

filtered = movingAverage(noised_samples, h)

plt.figure(figsize=(8, 4))
plt.plot(t, function,
         label="Funkcja oryginalna")
plt.plot(samples_t[h:], filtered, '--g',
         label=f"Pomiary uśrednione, h={h}")
plt.scatter(samples_t, noised_samples, 
            color="red", 
            marker='.',
            label=f"Pomiary zaszumione, noise_scale={noise_scale}")
plt.yticks(np.arange(-1.25, 1.25, 0.25))
plt.xticks(np.arange(0.0, 6.5, 0.5))
plt.xlabel("t")
plt.ylabel("f(t)")
plt.grid()
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(h_array, mse_array, '--o')
plt.title("Zależność MSE od wartośći h")
plt.xlabel("h")
plt.ylabel("MSE(h)")
plt.xticks(np.arange(0, h_max, 1))
plt.grid()
plt.show()
