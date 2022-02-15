################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 4 #
# 2/8/2022 #
# Convolution Part 2# 
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
        y[i] = np.exp(-2*t[i]) * (u(t[i]) - u(t[i]-3))
    return y #send back the output stored in an array

def f2(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = u(t[i] - 2) - u(t[i] - 6)
    return y #send back the output stored in an array

def f3(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(0.5*np.pi * t[i]) * u(t[i]) # 0.5 * pi = 0.25 Hz
    return y #send back the output stored in an array

# array scale version of u(t)
def u_forcing(function):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = u(t[i])
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


def f1_integral(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = 0.5 * (-np.exp(-2 * t[i]) + 1) * u(t[i]) - 0.5 * (-np.exp(-2 * t[i] - 3) + 1) * u(t[i] - 3)
    return y #send back the output stored in an array

def f2_integral(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = (t[i] - 2) * u(t[i] - 2) - (t[i] - 6) * u(t[i] - 6)
    return y #send back the output stored in an array

def f3_integral(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = (np.sin(0.5*np.pi * t[i]) * u(t[i]))/(0.5 * np.pi) # 0.5 * pi = 0.25 Hz
    return y #send back the output stored in an array


t = np.arange(-10, 10 + steps, steps)
t2 = np.arange(-20, 20 + steps/2, steps)

# Part 1: Impulse Response Functions

plt.figure(figsize = (30, 21))
y = f1(t)
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1')
plt.title('Part 1')

y = f2(t)
plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f2')
plt.title('Part 1')

y = f3(t)
plt.subplot(3, 1, 3)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f3')
plt.title('Part 1')
plt.show()

# Part 2: Convolutions of Step Response

plt.figure(figsize = (30, 21))
y = convolution(f1(t), u_forcing(t)) * steps
plt.subplot(3, 1, 1)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1 Step Response')
plt.title('Part 2')

y = convolution(f2(t), u_forcing(t)) * steps
plt.subplot(3, 1, 2)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f2 Step Response')
plt.title('Part 2')

y = convolution(f3(t), u_forcing(t)) * steps
plt.subplot(3, 1, 3)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f3 Step Response')
plt.title('Part 2')
plt.show()

# Part 3: Hand-Calculated Convolution Integrals

# this is different from the summation version of function 1
# python plots summation differently from integral version
plt.figure(figsize = (30, 21))
y = f1_integral(t)
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f1')
plt.title('Part 3')

# this is different from the summation version of function 2
# python plots summation differently from integral version
y = f2_integral(t)
plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f2')
plt.title('Part 3')

# same regardless of method used
y = f3_integral(t)
plt.subplot(3, 1, 3)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f3')
plt.title('Part 3')
plt.show()