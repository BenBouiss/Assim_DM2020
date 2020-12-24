
import numpy as np

def mycost(ival,x,y,norm):
    """
    compute value of cost function for linear regression
    """

    a=ival[0]  #slope
    b=ival[1]  #intercept
    
    #predicted value
    predicted_value = a*x + b

    #residues
    r = y - predicted_value
    
    # compute the cost / norm if residues
    # L1 norm
    if norm==1:
        val = np.sum(np.abs(r))
    
    # L2 norm 
    if norm==2:
        val = np.sqrt( np.sum(r**2) )
    
    #L_infinite norm
    if norm==3:
        val = max (np.abs(r)) 
    
    return val

def getGMatrix(x,order):
    '''
    Compute the G matrix needed to solve the polynomial regression problem
    by the least square solution
    y = G m 
    where Y are the ordinate data
    m the vector of the N+1 coeffcients of the polynomial 
    G the matrix containing:
    [ x^N x^N-1 .... 1 ]
    [ .................]
    
    '''
    Ndata=len(x)
    G=np.ones((Ndata,order+1))
    for i in range(0,order):
        G[:,i]=x**float(order-i)
    
    return G
    
    
    
    
    
    
    
    