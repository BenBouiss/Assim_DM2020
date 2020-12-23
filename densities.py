
import numpy as np

# Example 11.4
# from Parameter Estimation and Inverse Problems, 2nd edition, 2011
# by R. Aster, B. Borchers, C. Thurber
# lr=log(r(x,y))
# For this problem, we'll use a multivariate normal generator, with
# standard deviations specified by the vector step.

# Note that logproposal.m and generate.m
# are closely tied to each other.
def logproposal(x, y, step):

    lr = (-0.5)*np.sum((x-y)**2 / step**2)

    return lr

def generate(x, step):
    
    nparam = x.shape[0]
    y = x + step * np.random.randn(nparam)

    return y



