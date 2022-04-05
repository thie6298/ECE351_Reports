################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 10 #
# 3/29/2022 #
# Frequency Response # 
# #
################################################################

import numpy as np
import scipy.fftpack as ft
import scipy.signal as sig
import matplotlib.pyplot as plt
import control as con

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-7 # Define step size(lower for high frequency run)
log_steps = 1e2 # step size for logarithmic scale

R = 1000 # 1 kOhm
L = 0.027 # 27 mHenries
C = 0.0000001 # 100 nFarads

def x(t):
    # input signal function
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of f
        y[i] += np.cos(2*np.pi*100*t[i]) + np.cos(2*np.pi*3024*t[i]) + np.sin(2*np.pi*50000*t[i])
    return y

def H_dB(f, R, L, C):
    # transfer function magnitude
    y = np.zeros(f.shape) # initialize y(t) as an array of zeros
    for i in range(len(f)): # run the loop once for each index of f
        y[i] += (f[i]/(R*C))/(np.sqrt(f[i]**4 + ((1/(R*C))**2 - 2/(L*C)) * f[i]**2 + (1/(L*C))**2))
    for i in range(len(f)): # run the loop again to convert everything to dB 
        y[i] = 20 * np.log10(y[i])
    return y

def H_phase(f, R, L, C):
    #transfer function phase
    y = np.zeros(f.shape) # initialize y(t) as an array of zeros
    for i in range(len(f)): # run the loop once for each index of f
        y[i] += np.pi/2 - np.arctan((f[i]/(R*C))/(-(f[i]**2) + 1/(L*C)))
    for i in range(len(f)): # run the loop again to convert to degrees & correct the graph
        y[i] = 180 * y[i] / np.pi
        if y[i] > 90:
            y[i] = y[i] - 180
    return y

# PART 1
plt.figure(figsize = (30, 21))
f = np.arange(1e3, 1e6 + log_steps, log_steps)
    
plt.subplot(2, 1, 1)
y = H_dB(f, R, L, C)
plt.semilogx(f, y)
plt.grid()
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function(hand calculated)')

plt.subplot(2, 1, 2)
y = H_phase(f, R, L, C)
plt.semilogx(f, y)
plt.grid()
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function(hand calculated)')
    
plt.show()

plt.figure(figsize = (30, 21))
bodeNum = [1/(R*C), 0]
bodeDen = [1, 1/(R*C), 1/(L*C)]
w, mag, phase = sig.bode((bodeNum, bodeDen), f)

plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.grid()
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function(sig.bode)')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function(sig.bode)')

plt.show()

f2 = np.arange(np.pi*2e3, np.pi*2e6 + log_steps, log_steps)
sys = con.TransferFunction(bodeNum, bodeDen)
_ = con.bode(sys, f2, dB = True , Hz = True , deg = True , Plot = True)
# con graphs automatically without needing to use plt

# PART 2
plt.figure(figsize = (30, 21))
t = np.arange(0, 1e-2 + steps, steps)

plt.subplot(2, 1, 1)
x_in = x(t)
plt.plot(t, x_in)
plt.grid()
plt.xlabel('Time(s)')
plt.ylabel('Input Signal')
plt.title('Input Signal')

#analog to digital transformation
ZNum, ZDen = sig.bilinear(bodeNum, bodeDen, 1)

plt.subplot(2, 1, 2)
y_out = sig.lfilter(ZNum, ZDen, x_in)
plt.plot(t, y_out)
plt.grid()
plt.xlabel('Time(s)')
plt.ylabel('Output Signal')
plt.title('Output Signal')

plt.show()


