################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 12 #
# 4/26/2022 #
# Filter Project # 
# #
################################################################

import numpy as np
import scipy.signal as sig
import scipy.fftpack as ft
import matplotlib.pyplot as plt
import pandas as pd

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

def bandpass_magnitude(freq):
    # pass range 1.8kHz to 2.0 kHz
    # R = 50, L = 7m, C = 1u
    y = np.zeros(freq.shape)
    # convert to imaginary and rad/sec for Fourier domain
    for i in range(len(freq)):
        jw = 2 * np.pi * freq[i] * 1j
        y[i] = np.abs(50 * jw / (7e-3 * jw**2 + 50 * jw + 1e6))
    return y

def make_stem(ax ,x,y,color='k',style='solid',label='',linewidths =2.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths=linewidths)
    ax.set_ylim ([1.05*y.min(), 1.05*y.max()])
    
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

# initial(noise) plots
fig, ax = plt.subplots(figsize =(10, 7))
freq, Xmag, Xphi = fft_routine2(sensor_sig, 1e6)

make_stem(ax, freq, Xmag)
plt.title('Noisy Input Signal')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 1e5)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag)
plt.title('Noisy Input Signal(Full View)')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 5e5)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag)
plt.title('Low Frequency Noise')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 1e3)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag)
plt.title('Target Frequencies for Band Pass')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1e3, 3e3)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag)
plt.title('High Frequency Noise')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(3e3, 5e5)
plt.show()


# bode plots
bodeNum = [50, 0]
bodeDen = [7e-3, 50, 1e6]

f = np.arange(1e1, 7e6 + 1, 1)
w, mag, phase = sig.bode((bodeNum, bodeDen), f)
freq = w / (2 * np.pi)

plt.figure(figsize = (30, 21))
plt.subplot(2, 1, 1)
plt.semilogx(freq, mag)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function')

plt.subplot(2, 1, 2)
plt.semilogx(freq, phase)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function')
plt.show()

f = np.arange(1e0, 7e3 + 0.01, 0.01)
w, mag, phase = sig.bode((bodeNum, bodeDen), f)
freq = w / (2 * np.pi)

plt.figure(figsize = (30, 21))
plt.subplot(2, 1, 1)
plt.semilogx(freq, mag)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function(Low Frequencies)')

plt.subplot(2, 1, 2)
plt.semilogx(freq, phase)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function(Low Frequencies)')
plt.show()

f = np.arange(7e3, 2e4 + 0.1, 0.1)
w, mag, phase = sig.bode((bodeNum, bodeDen), f)
freq = w / (2 * np.pi)

plt.figure(figsize = (30, 21))
plt.subplot(2, 1, 1)
plt.semilogx(freq, mag)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function(Band Pass)')

plt.subplot(2, 1, 2)
plt.semilogx(freq, phase)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function(Band Pass)')
plt.show()

f = np.arange(2e4, 7e6 + 1, 1)
w, mag, phase = sig.bode((bodeNum, bodeDen), f)
freq = w / (2 * np.pi)

plt.figure(figsize = (30, 21))
plt.subplot(2, 1, 1)
plt.semilogx(freq, mag)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude(dB)')
plt.title('Transfer Function(High Frequencies)')

plt.subplot(2, 1, 2)
plt.semilogx(freq, phase)
plt.grid()
plt.xlabel('Frequency(Hz)')
plt.ylabel('Phase Angle(degrees)')
plt.title('Transfer Function(High Frequencies)')
plt.show()



# filtered plots
fig, ax = plt.subplots(figsize =(10, 7))
freq, Xmag, Xphi = fft_routine2(sensor_sig, 1e6)

make_stem(ax, freq, Xmag * np.abs(bandpass_magnitude(freq)))
plt.title('Filtered Signal')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 1e5)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag * np.abs(bandpass_magnitude(freq)))
plt.title('Filtered Signal(Full View)')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 5e5)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag * np.abs(bandpass_magnitude(freq)))
plt.title('Filtered Low Frequency')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1, 1e3)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag * np.abs(bandpass_magnitude(freq)))
plt.title('Filtered Band Pass')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(1e3, 3e3)
plt.show()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, Xmag * np.abs(bandpass_magnitude(freq)))
plt.title('Filtered High Frequency')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Voltage(V)')
plt.xlim(3e3, 5e5)
plt.show()