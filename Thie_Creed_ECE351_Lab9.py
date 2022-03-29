################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 9 #
# 3/22/2022 #
# Fast Fourier Transform # 
# #
################################################################

import numpy as np
import scipy.fftpack as ft
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size

# background functions
def b_k(k):
    return (2/(k*np.pi)) * (1-np.cos(k*np.pi)) # for f4

# graphed functions
def f1(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(2 * np.pi * t[i])
    return y

def f2(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = 5 * np.sin(2 * np.pi * t[i])
    return y

def f3(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = 2 * np.cos(4 * np.pi * t[i] - 2) + np.sin(12 * np.pi * t[i] + 3)**2 
    return y

def f4(t):
    # Fourier series from Lab 8, using preset k and T values
    k = 15
    T = 8
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        for j in range(1, k+1): # run the loop for each number in k to get the total Fourier sum
            y[i] += b_k(j) * np.sin((j*2*np.pi*t[i])/T)
    return y

# base fft routine
def fft_routine(x, fs):
    N = len(x) # find the length of the signal
    X_fft = ft.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = ft.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    return freq, X_mag, X_phi

# cleaned fft routine
def fft_routine2(x, fs):
    N = len(x) # find the length of the signal
    X_fft = ft.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = ft.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal

    for i in range (0, len(X_mag)):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
            
    return freq, X_mag, X_phi

# graph the fft routine
for x in f1, f2, f3:    
    plt.figure(figsize = (30, 21))
    
    t = np.arange(0, 2, steps)
    freq, X_mag, X_phi = fft_routine(x(t), 10)
    
    plt.subplot(3, 1, 1)
    y = x(t)
    plt.plot(t, y)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Base Function')
    
    plt.subplot(3, 2, 3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('abs(X)')
    plt.title('FFT Magnitude')
   
    plt.subplot(3, 2, 6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('angle(X)')
    plt.title('FFT Phase')
    plt.xlim(-2, 2)
    
    plt.subplot(3, 2, 4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('abs(X)')
    plt.title('FFT Magnitude')  
    plt.xlim(-2, 2)
    
    plt.subplot(3, 2, 5)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('angle(X)')
    plt.title('FFT Phase')
    plt.show()

# graph the cleaned fft routine
for x in f1, f2, f3, f4:    
    plt.figure(figsize = (30, 21))

    if x == f4:
        t = np.arange(0, 16, steps)
    else:
        t = np.arange(0, 2, steps)

    freq, X_mag, X_phi = fft_routine2(x(t), 10000)
    
    plt.subplot(3, 1, 1)
    y = x(t)
    plt.plot(t, y)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Base Function')
    
    plt.subplot(3, 2, 3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('abs(X)')
    plt.title('FFT Magnitude')
   
    plt.subplot(3, 2, 6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('angle(X)')
    plt.title('FFT Phase')
    if x == f3:
        plt.xlim(-15,15)
    else:
        plt.xlim(-2, 2)
    
    plt.subplot(3, 2, 4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('abs(X)')
    plt.title('FFT Magnitude')
    if x == f3:
        plt.xlim(-15,15)
    else:
        plt.xlim(-2, 2)

    plt.subplot(3, 2, 5)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlabel('f(Hz)')
    plt.ylabel('angle(X)')
    plt.title('FFT Phase')
    plt.show()
    




