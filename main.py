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

h = 30
t = np.linspace(0, 2*np.pi, 628)
function = np.sin(t)
noise = np.random.randn(t.size) / 10
noised_function = function + noise
filtered = movingAverage(noised_function, h)

plt.plot(t, noised_function)
plt.plot(t, function)
plt.plot(t[h:], filtered)
plt.show()
