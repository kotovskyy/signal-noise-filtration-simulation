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
        raise ValueError(f"Number of samples in first and second arrays must match! (first={(first.size)}, second={(second.size)})")
    return np.mean(np.square(first-second))
    

noise_scale = 0.15
h_array = np.arange(1, 21, 1)  
t = np.linspace(0, 2*np.pi, 1000)
function = np.sin(t)

n_samples = 250
samples_scale = int(t.size / n_samples)
samples_array = np.sin(t[::samples_scale])


noise = noise_scale * np.random.randn(n_samples) 
noised_function = samples_array + noise
mse_array = np.zeros(h_array.size)

for h in h_array:
    filtered = movingAverage(noised_function, h)
    mse_array[h-1] = MSE(filtered, function[h*samples_scale::samples_scale])

# h = 7
# filtered = movingAverage(samples_array, h)
# print(f"FILTERED SIZE : {filtered.size}")
# print

# plt.scatter(t[::samples_scale], noised_function)
# plt.plot(t, function)
# plt.plot(t[h:], filtered)
min_mse = np.argmin(mse_array)
print(f"MIN MSE = {mse_array[min_mse]}, h = {h_array[min_mse]}")


plt.scatter(t[::samples_scale], noised_function, color="red", marker='.')
plt.plot(t, function)
plt.plot(t[5*samples_scale::samples_scale], movingAverage(noised_function, 5)) 
# plt.plot(h_array, mse_array)
plt.show()
