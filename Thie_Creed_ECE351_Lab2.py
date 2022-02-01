################################################################
# #
# Creed Thie #
# ECE 351-53 #
# Lab 2 #
# 1/25/2022 #
# Graph math functions from numpy using plt # 
# #
################################################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14 # Set font size in plots

steps = 1e-2 # Define step size

def u(t):
    return 1 if t >= 0 else 0
    
def r(t):
    return t if t >= 0 else 0

# Part 1
def PlotCos(t): # The only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = np.cos(t[i])
    return y #send back the output stored in an array

# Parts 2 and 3
def PlotStepRampCombination(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = r(t[i]) - r(t[i]-3) + 5 * u(t[i]-3) - 2 * u(t[i]-6) - 2 * r(t[i]-6)
    return y #send back the output stored in an array

# Part 3 Derivative
def PlotStepRampSlope(t):
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    for i in range(len(t)): # run the loop once for each index of t
        y[i] = u(t[i]) - u(t[i]-3) - 2 * u(t[i]-6)
    return y #send back the output stored in an array

# cosine
t1 = np.arange(0, 10 + steps , steps)
y1 = PlotCos(t1)

plt.figure(figsize = (30, 21))
plt.subplot(5, 2, 1)
plt.plot(t1, y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('cos(t)')
plt.title('Part 1')

# custom function
t2 = np.arange(-5, 10 + steps , steps)
y2 = PlotStepRampCombination(t2)

plt.subplot(5, 2, 2)
plt.plot(t2, y2)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 2')

# time reversal
t3 = np.arange(-10, 5 + steps , steps)
y3 = PlotStepRampCombination(-t3)

plt.subplot(5, 2, 3)
plt.plot(t3, y3)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 3.1(Reversal)')

#time shift 1
t4 = np.arange(-1, 14 + steps , steps)
y4 = PlotStepRampCombination(t4-4)

plt.subplot(5, 2, 4)
plt.plot(t4, y4)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 3.2(Shift)')

#time shift 2
t5 = np.arange(-14, 1 + steps , steps)
y5 = PlotStepRampCombination((-t5)-4)

plt.subplot(5, 2, 5)
plt.plot(t5, y5)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 3.2(Shift with Reversal)')

# time half rescale
t6 = np.arange(-10, 20 + steps , steps)
y6 = PlotStepRampCombination(t6/2)

plt.subplot(5, 2, 6)
plt.plot(t6, y6)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 3.3(Halved)')

# time double rescale
t7 = np.arange(-5, 10 + steps , steps)
y7 = PlotStepRampCombination(t7 * 2)

plt.subplot(5, 2, 7)
plt.plot(t7, y7)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Part 3.3(Doubled)')

# derivative(hand calculated)
t8 = np.arange(-5, 10 + steps , steps)
y8 = PlotStepRampSlope(t8)

plt.subplot(5, 2, 8)
plt.plot(t8, y8)
plt.grid()
plt.xlabel('t')
plt.ylabel('f`(t)')
plt.title('Part 3.4(Derivative)')

# derivative(numpy)
steps = 1e-2 # Define step size
t9 = np.arange(-5, 10 + steps , steps)
dt = np.diff(t9)
dy = np.diff(PlotStepRampCombination(t9), axis = 0)/dt
print(PlotStepRampCombination(t9))
print(dy)

plt.subplot(5, 2, 9)
plt.plot(t9[range(len(dy))],dy)
plt.grid()
plt.xlabel('t')
plt.ylabel('f`(t)')
plt.title('Part 3.5(Derivative)')
plt.show()