################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 7 #
# 3/1/2022 #
# Block Diagrams and System Stability # 
# #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size

# numerators and denominators
numG = [1, 9]
denG = [1, -2, -40, -64]
numA = [1, 4]
denA = [1, 4, 3]
numB = [1, 26, 168]

# background calc functions
def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

# PART 1
#roots and poles
print(sig.tf2zpk(numG, denG))
print(sig.tf2zpk(numA, denA))
print(np.roots(numB))

#open loop
numHo = sig.convolve(numA, numG)
denHo = sig.convolve(denA, denG)

t = np.arange(0, 10 + steps, steps)

plt.figure(figsize = (30, 21))
t, Ho = sig.step((numHo, denHo), T=t)
plt.subplot(1, 1, 1)
plt.plot(t, Ho)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response(Open Loop Transfer Function)')
plt.show()

# PART 2

numHt = sig.convolve(numA, numG)
denHt = sig.convolve(numB, sig.convolve(denA, denG)) + np.pad(sig.convolve(denA, denG), (2, 0), 'constant', constant_values=(0,0))

print(sig.tf2zpk(numHt, denHt))

plt.figure(figsize = (30, 21))
t, Ht = sig.step((numHt, denHt), T=t)
plt.subplot(1, 1, 1)
plt.plot(t, Ht)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response(Closed Loop Transfer Function)')
plt.show()




