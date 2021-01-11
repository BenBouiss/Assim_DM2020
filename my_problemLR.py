import numpy as np

Xi = None
sigmad = None
Yi = None

def logprior(m):
    '''
    A priori probability on paramaters
    :param m:
    :return: log(prior)
    '''
    
    # no prior knowledge about the intercept or the slope, it can be any value!
    lp = 0

    return lp


def loglikelyhood(m, ind):
    '''
    given a set of parameter m, compute the probalility that
    it explains the data
    :param m:
    :return:
    '''

    #
    # proposed values
    #
    # slope 
    a = m[0]
    # intercept 
    b = m[1]
    
    Tcal = a * Xi + b
    #
    # compute residuals
    #
    residuals = (Yi-Tcal)/sigmad 

    #
    # compute likelyhood
    #
    # log (gaussian or exponential or whatever)
    #l = ??
    # log (normal density for the residue)
    if ind == 'Gaus':
        l = - 0.5 * np.sum(residuals ** 2) #GAUSSIAN
    elif ind == 'Expo':
        l = -1*np.sum(np.abs(residuals)) #EXP
    else:
        print('Distribution not implemented')
        return 

    return l
