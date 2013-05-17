
"""
Polynomial interpolation.
Modified by: Peixiang Xu
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi,yi):
    """
    Quadratic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(3),xi,xi**2]).T
    b = yi
    c = solve(A,b)
    return c


def test_quad1():
    """
    Test code of quar_interp, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2.])
    yi = np.array([ 1., -1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def test_quad2():
    """
    Test code of quar_interp, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2.])
    yi = np.array([ 1., 1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([1.,  1.,  1.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def plot_quad(xi,yi):
    """
    Plot the quadratic interpolating and the data points, and
    saves as a png.

    """
    c = quad_interp(xi,yi)
    x = linspace(xi.min() - 1, xi.max(), 1000)
    y = c[0] + c[1]*x + c[2]*x**2
    
    plt.figure(1)
    plt.clf()
    plt.plot(x,y,'r-')
    plt.plot(xi,yi,'bo')

    plt.ylim(yi.min()-1,yi.max()+1)
    plt.savefig('quadratic.png')


def cubic_interp(xi,yi):
    """
    Cubic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 4"
    assert len(xi)==4 and len(yi)==4, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(4),xi,xi**2,xi**3]).T
    b = yi
    c = solve(A,b)
    return c

def test_cubic1():
    """
    Test code of cubic_interp, no return value or exception if test runs properly.
    """
    xi = np.array([-1., 0., 2., 1.])
    yi = np.array([-2., 1., 1., 0.])
    c = cubic_interp(xi,yi)
    c_true = np.array([1, 0, -2, 1])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def plot_cubic(xi,yi):
    """
    Plot the cubic interpolating  and the data points, and
    saves as a png.

    """
    c = quad_interp(xi,yi)
    x = linspace(xi.min() - 1, xi.max(), 1000)
    y = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
    
    plt.figure(1)
    plt.clf()
    plt.plot(x,y,'r-')
    plt.plot(xi,yi,'bo')

    plt.ylim(yi.min()-1,yi.max()+1)
    plt.savefig('cubic.png')

def poly_interp(xi,yi):
    """
    Polynomial interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,...n-1, n is the
    length of xi and yi.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + ... + c[n-1]*x**(n-1).

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have equal length"
    assert len(xi) == len(yi), error_message

    n = len(xi)
    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(n)] + [xi**k for k in range(1,n)]).T
    b = yi
    c = solve(A,b)
    return c

def test_poly1():
    """
    Test code of poly_interp, no return value or exception if test runs properly.
    """
    xi = np.array([-1., 0., 2., 1.])
    yi = np.array([-2., 1., 1., 0.])
    c = poly_interp(xi,yi)
    c_true = np.array([1, 0, -2, 1])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def plot_poly(xi,yi):
    """
    Plot the polynomial interpolating and the data points, and
    saves as a png.

    """
    c = quad_interp(xi,yi)
    x = linspace(xi.min() - 1, xi.max(), 1000)
    y = c[n-1]
    for i in range(n-1,0,-1):
        y = y * x + c[i-1]
    
    plt.figure(1)
    plt.clf()
    plt.plot(x,y,'r-')
    plt.plot(xi,yi,'bo')

    plt.ylim(yi.min()-1,yi.max()+1)
    plt.savefig('poly.png')

if __name__=="__main__":
    # "main program"
    # the code below is executed only if the module is executed at the command line,
    #    $ python hw2b.py
    # or run from within Python, e.g. in IPython with
    #    In[ ]:  run hw2b
    # not if the module is imported.
    print "Running test..."
    test_quad1()
    test_quad2()
    test_cubic1()
    test_poly1()
