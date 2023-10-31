# Simulation of signal samples filtration
This project was created as a part of 'Adaptive control' (pol. 'Sterowanie adaptacyjne) course. 
It's main goal is a simulation of how we can reduce noise applied to signal's samples. In this 
case, the signal is a `sin(t)` and every sample has a noise randomly generated from normal dis-
tribution with given variancy. 
Three main tasks for this projects are:
  1) How MSE error between original signal `sin(t)` and our samples depends on variable `h` - number of previous samples considered in **moving average** filter to calculate new sample's value.
  2) How MSE depends on `Var(Z)` - variance of the noise applied to samples.
  3) Find best `h` values for different variances.

For more information, I recommend you to look at <a href="https://github.com/kotovskyy/signal-noise-filtration-simulation/blob/master/report.pdf">report.pdf</a> (Polish language only).
