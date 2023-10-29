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
    

noise_scale = 0.2
h_array = np.arange(1, 61, 1)  
t = np.linspace(0, 2*np.pi, 628)
function = np.sin(t)
noise = noise_scale * np.random.randn(t.size) 
noised_function = function + noise
mse_array = np.zeros(h_array.size)

for h in h_array:
    filtered = movingAverage(noised_function, h)
    mse_array[h-1] = MSE(filtered, function[h:])

min_mse = np.argmin(mse_array)
print(f"MIN MSE = {mse_array[min_mse]}, h = {min_mse+1}")

# plt.plot(t, noised_function)
# plt.plot(t, function)
# plt.plot(t[h:], filtered)
plt.plot(h_array, mse_array)
plt.show()
