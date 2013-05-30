"""
Copyright (c) 2013, Peixiang Xu(peixiangxu@gmail.com)
 
This program is free software: you can redistribute it 
and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation.
"""

"""
Quadratic interpolation.
Modified by: Peixiang Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

# Set up linear system to interpolate through data points:

# Data points:
xi = np.array([-1., 0., 2])
yi = np.array([1., -1., 7.])

# It would be better to define A in terms of the xi points.
# Doing this is part of the homework assignment.
A = np.array([[1., xi[0], xi[0]**2], [1., xi[1], xi[1]**2], [1., xi[2], \
    xi[2]**2]])
b = yi

# Solve the system:
c = solve(A,b)

print "The polynomial coefficients are:"
print c

# Plot the resulting polynomial:
x = np.linspace(-2,3,1001)   # points to evaluate polynomial
y = c[0] + c[1]*x + c[2]*x**2

plt.figure(1)       # open plot figure window
plt.clf()           # clear figure
plt.plot(x,y,'b-')  # connect points with a blue line

# Add data points  (polynomial should go through these points!)
plt.plot(xi,yi,'ro')   # plot as red circles
plt.ylim(-2,8)         # set limits in y for plot

plt.title("Data points and interpolating polynomial")

plt.savefig('hw2a.png')   # save figure as .png file
