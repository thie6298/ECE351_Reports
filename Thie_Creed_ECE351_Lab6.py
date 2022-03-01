################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 6 #
# 2/22/2022 #
# Partial Fraction Expansion # 
# #
################################################################

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size

# background calc functions
def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

# step response for the first equation
def step_response_y(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = (0.5 + np.exp(-6 * t[i]) - 0.5 * np.exp(-4 * t[i])) * u(t[i])
    return y #send back the output stored in an array

# cosine method for the second equation
def cosine_method(t, roots, poles): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    root_count = 0
    for root in roots:
        k = np.abs(root)
        k_angle = np.angle(root)
        alpha = poles[root_count][0]
        omega = poles[root_count][1]
        for i in range(len(t)): # run the loop once for each index of t
            y[i] += (k * np.exp(alpha * t[i]) * np.cos(omega * t[i] + k_angle)) * u(t[i])
        root_count += 1
    return y #send back the output stored in an array

# sine method for the second equation
def sine_method(t, poles): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for pole in poles:
        p = complex(pole[0], pole[1])
        # non-complex part of the equation is 25250 / (s * (s + 10))
        g = 25250 / (p * (p + 10))
        g_abs = np.abs(g)
        g_angle = np.angle(g)
        for i in range(len(t)): # run the loop once for each index of t
            y[i] += (g_abs/pole[1]) * np.exp(pole[0] * t[i]) * np.sin(pole[1] * t[i] + g_angle) * u(t[i])
    return y #send back the output stored in an array

t = np.arange(0, 10 + steps, steps)
t3 = np.arange(0, 4.5 + steps, steps)

# Part 1

plt.figure(figsize = (30, 21))
y = step_response_y(t)
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response(Partial Fraction Expansion)')
plt.show()

num1 = [1, 6, 12]
den1 = [1, 10 ,24]

plt.figure(figsize = (30, 21))
t2, y = sig.step((num1, den1), T = t)
plt.subplot(1, 1, 1)
plt.plot(t2, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Step Response(scipy)')
plt.show()

# use partial fraction expansion of Y(s) to check answer
num2 = [1, 6, 12]
den2 = [1, 10 ,24, 0]
pfe2 = sig.residue(num2, den2)
print(pfe2)

# Part 2

# Y(s) = 25250 / (s**6 + 18*s**5 + 218*s**4 + 2036*s**3 + 9085*s**2 + 25250*s)
num3 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250, 0]
pfe3 = sig.residue(num3, den3)
print(pfe3)

# sine method complex conjugate p values are -3 + 4j and -1 + 10j
plt.figure(figsize = (30, 21))
y = sine_method(t3, [[-3, 4], [-1, 10]])
plt.subplot(1, 1, 1)
plt.plot(t3, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Complex Step Response(Sine Method)')
plt.show()

# cosine method using r and p values
plt.figure(figsize = (30, 21))
y = cosine_method(t3, [1, -0.48557692+0.72836538j, -0.48557692-0.72836538j, -0.21461963, 0.09288674-0.04765193j, 0.09288674+0.04765193j],\
                         [[0, 0], [-3, 4], [-3, -4], [-10, 0], [-1, 10], [-1, -10]])
plt.subplot(1, 1, 1)
plt.plot(t3, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Complex Step Response(Cosine Method)')
plt.show()

num4 = [25250]
den4 = [1, 18, 218, 2036, 9085, 25250]

plt.figure(figsize = (30, 21))
t4, y = sig.step((num4, den4), T = t3)
plt.subplot(1, 1, 1)
plt.plot(t4, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Complex Step Response(scipy)')
plt.show()