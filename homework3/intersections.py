# Newton Method to calculate intersections

from newton import solve
import numpy as np
import matplotlib.pyplot as plt
import math

def fvals(x):
    """
    Return f(x) and f'(x) for applying Newton to find intersections.
    """
    f = 1 - 0.6*x**2 - x*math.cos(math.pi*x)
    fp = x*math.pi*math.sin(x*math.pi) - 1.2*x - math.cos(math.pi*x)
    return f, fp

def main(debug = False):
    x = np.linspace(-5,5,100)
    y1 = [x1 * math.cos(x1 * math.pi) for x1 in x]
    y2 = 1 - 0.6*x**2

    plt.figure(1)
    plt.clf()
    plt.plot(x,y1,label='g1')
    plt.plot(x,y2,label='g2')

    guesses = [-2,-1.5,-0.8,1.5]
    for x0 in guesses:
        x,iter = solve(fvals,x0,debug)
        print 'With initial guess x0 = %22.15e' %x0
        print '\tsolve return x = %22.15e after %d iterations' %(x,iter)
        plt.plot(x,1 - 0.6*x**2,'k.')
    plt.legend(('g1','g2'))
    plt.show()
    plt.savefig('intersections.png') 

if __name__ == "__main__":
    main()
