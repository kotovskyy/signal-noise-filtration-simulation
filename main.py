import numpy as np
from matplotlib import pyplot as plt

t = np.linspace(0, 2*np.pi, 628)
function = np.sin(t)
noise = np.random.randn(t.size) / 10
noised_function = function + noise

plt.plot(t, noised_function)
plt.plot(t, function)
plt.show()