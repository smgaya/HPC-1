"""
Copyright (c) 2013, Peixiang Xu(peixiangxu@gmail.com)
 
This program is free software: you can redistribute it 
and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation.
"""

# Newton Method to calculate square root

def fvals_sqrt(x):
    """
    Return f(x) and f'(x) for applying Newton to find a square root.
    """
    f = x**2 - 4.
    fp = 2.*x
    return f, fp

def solve(fvals,x0,debug = False):
    if debug:
        print "Initial guess: x = %22.15e" % x0
    maxiter = 20
    tol = 1e-14
    iter = 0

    while iter <= maxiter and abs(fvals(x0)[0] - 0) > tol:
        x0 = x0 - fvals(x0)[0]/fvals(x0)[1]
        iter += 1
        if debug:
            print "After %d iterations, x = %22.15e" % (iter,x0)
    return x0,iter

def test1(debug_solve=False):
    """
    Test Newton iteration for the square root with different initial
    conditions.
    """
    from numpy import sqrt
    for x0 in [1., 2., 100.]:
        print " "  # blank line
        x,iters = solve(fvals_sqrt, x0, debug=debug_solve)
        print "solve returns x = %22.15e after %i iterations " % (x,iters)
        fx,fpx = fvals_sqrt(x)
        print "the value of f(x) is %22.15e" % fx
        assert abs(x-2.) < 1e-14, "*** Unexpected result: x = %22.15e"  % x
