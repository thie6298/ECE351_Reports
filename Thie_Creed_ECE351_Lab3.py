################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 3 #
# 2/1/2022 #
# Convolution # 
# #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size

def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

def f1(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = u(t[i]-2) - u(t[i]-9)
    return y #send back the output stored in an array

def f2(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.exp(-t[i]) * u(t[i])
    return y #send back the output stored in an array

def f3(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = r(t[i]-2) * (u(t[i]-2) - u(t[i]-3)) + r(4-t[i]) * (u(t[i]-3) - u(t[i]-4))
    return y #send back the output stored in an array

def convolution(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 - 1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 - 1)))
    #.shape means same length as f1Extended
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if (i-j+1 > 0):
                result[i] += f1Extended[j] * f2Extended[i-j+1]
    return result

# Part 1
t = np.arange(0, 20 + steps , steps)
t2 = np.arange(0, 40 + 2 * steps, steps)
''' PART 1
plt.figure(figsize = (30, 21))

y1 = f1(t)
plt.subplot(3, 1, 1)
plt.plot(t, y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1(t)')
plt.title('Part 1')

y2 = f2(t)
plt.subplot(3, 1, 2)
plt.plot(t, y2)
plt.grid()
plt.xlabel('t')
plt.ylabel('f2(t)')

y3 = f3(t)
plt.subplot(3, 1, 3)
plt.plot(t, y3)
plt.grid()
plt.xlabel('t')
plt.ylabel('f3(t)')
plt.show()'''

plt.figure(figsize = (30, 21))
y = convolution(f1(t), f2(t)) * steps
plt.subplot(3, 1, 1)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1 * f2')
plt.title('Part 2')

y = convolution(f1(t), f3(t)) * steps
plt.subplot(3, 1, 2)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1 * f3')
plt.title('Part 2')

y = convolution(f2(t), f3(t)) * steps
plt.subplot(3, 1, 3)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f2 * f2')
plt.title('Part 2')
plt.show()

''' CHECK WORK
plt.figure(figsize = (30, 21))
y = sig.convolve(f1(t), f2(t)) * steps
plt.subplot(1, 1, 1)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1 * f2')
plt.title('Part 2')
plt.show()'''