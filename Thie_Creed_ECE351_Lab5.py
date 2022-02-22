################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 5 #
# 2/15/2022 #
# RLC Circuit Impulse Response# 
# #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-5 # Define step size

# background calc functions
def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

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

# sine method substeps
def sin_method_alpha(R, L, C):
    return -1/(2*R*C)

def sin_method_omega(R, L, C):
    return 0.5 * np.sqrt((1/(R*C))**2 - 4*(1/(np.sqrt(L*C)))**2 + 0*1j)

# calculate the impulse response using Laplace transform + sine method
def h_sine_method(t, R, L, C): # The only variable sent to the function is t
    alpha = sin_method_alpha(R, L, C)
    omega = sin_method_omega(R, L, C)
    p = alpha + omega
    g = 1/(R*C)*p
    g_abs = np.abs(g)
    g_angle = np.angle(g)
    y = np.zeros(t.shape) # initialize g(t) as an array of zeros
    for i in range(len(t)): # run the loop once for[ each index of t
        y[i] = (g_abs/np.abs(omega)) * np.exp(alpha * t[i]) * np.sin(np.abs(omega) * t[i] + g_angle) * u(t[i])
    return y #send back the output stored in an array

def h_scipy_method(t, R, L, C):
    num = [0, 1/(R*C), 0]
    den = [1, 1/(R*C), (1/np.sqrt(L*C))**2]
    return sig.impulse((num, den), T = t)

def h_step_response(t, R, L, C):
    num = [0, 1/(R*C), 0]
    den = [1, 1/(R*C), (1/np.sqrt(L*C))**2]
    return sig.step((num, den), T = t)

t = np.arange(0, 0.0012 + steps, steps)
R = 1000
L = 0.027
C = 0.0000001

plt.figure(figsize = (30, 21))
y = h_sine_method(t, R, L, C)
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Sine Method')
plt.show()

plt.figure(figsize = (30, 21))
t2, y = h_scipy_method(t, R, L, C)
plt.subplot(1, 1, 1)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Scipy Method')
plt.show()

plt.figure(figsize = (30, 21))
t3, y = h_step_response(t, R, L, C)
plt.subplot(1, 1, 1)
plt.plot(t3, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Step Response')
plt.show()

