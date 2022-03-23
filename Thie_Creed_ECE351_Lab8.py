################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 8 #
# 3/8/2022 #
# Fourier Series # 
# #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size
T = 8 #define length of time for Fourier series

# background calc functions
def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

def a_k(k):
    return 0 # always 0 because it must be the sine function of a whole number * pi
    
def b_k(k):
    return (2/(k*np.pi)) * (1-np.cos(k*np.pi)) # result of integrating to find b_k

# Fourier Series function constructing x(t) as a square wave
def x(t, k):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        for j in range(1, k+1): # run the loop for each number in k to get the total Fourier sum
            y[i] += b_k(j) * np.sin((j*2*np.pi*t[i])/T)
    return y

# PART 1
#test Fourier series terms
print(a_k(0))
print(a_k(1))
print(b_k(1))
print(b_k(2))
print(b_k(3))

t = np.arange(0, 20 + steps, steps)

# Fourier series at different resolutions
k = 1
plt.figure(figsize = (30, 21))
y = x(t, k)
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=1')

k = 3
y = x(t, k)
plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=3')

k = 15
y = x(t, k)
plt.subplot(3, 1, 3)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=15')
plt.show()

k = 50
plt.figure(figsize = (30, 21))
y = x(t, k)
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=50')

k = 150
y = x(t, k)
plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=150')

k = 1500
y = x(t, k)
plt.subplot(3, 1, 3)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('k=1500')
plt.show()





