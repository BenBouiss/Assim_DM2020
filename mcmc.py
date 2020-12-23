
import numpy as np


def mcmc(logprior, loglikelyhood, generate, logproposal, m0, niter, step):
    '''
    The Four first arguments are functions that return a probability
    
    logprior:      prior (before knowing data) density function of parameter
    loglikelyhood: datalikelyhood = probability of having <d> data value given <m> parameter
    generate: :    generate randomly a parameter vector hypothesis given the previous
                   parameter hypothesis
    logproposal:   r(x',x) probability for a new parameter hypothesis w/r previous
    
    m0:            initial parameter hypothesis
    niter:         total number of iteration (or markov chain steps)
    step:          the average (std) motion step for parameter
    
    return:        mout= all accepted solutions
                   mMAP= solution corresponding to maximum a posteriori probability
                   acceptance_rate
    '''
    #
    # Figure out some size information.
    #
    n = m0.shape[0]
    #
    # Allocate space for the results.
    #
    mout = np.zeros((niter, n))
    #
    # Initialize the starting point.
    #
    mout[0,:] = m0
    current = m0
    lMAP = -np.infty
    mMAP = current
    nacc = 0
    #
    # The main loop.
    #
    #
    for k in range(1, niter):
        # Generate a candidate model from the previous model
        # with normal (gaussian) probability density
        candidate = generate(current, step)

        #
        # Evaluate the logarithm of the acceptance ratio.
        #
        # logprior: log of a priori density on parameter: rho(y)
        # loglikelihood: log of density for the direct model f(d-g(m)): sigma(d,y)
        # logproposal: r(x,y)
        lpcandidate = logprior(candidate)
        llcandidate = loglikelyhood(candidate)

        # when r() is symetric, this step is useless since r(x,y)=r(y,x)
        lr2 = logproposal(current, candidate, step)
        lr1 = logproposal(candidate, current, step)

        lpcurrent = logprior(current)
        llcurrent = loglikelyhood(current)

        # A(x',x) = alpha = (p(y) * f(d, y) * r(y, x)) / (p(x) * f(d, x) * r(x, y))
        logalpha = lpcandidate + llcandidate + lr1 - lpcurrent - llcurrent - lr2

        #
        # Take the minimum of the log(alpha) and 0.
        #
        if logalpha > 0:
            logalpha = 0
        #
        # Generate a U(0,1) random number and take its logarithm.
        #
        logt = np.log(np.random.rand())
        #
        # Accept or reject the step.
        #
        if logt < logalpha:
            #
            # Accept the step.
            #
            current = candidate
            nacc = nacc + 1

        #
        # Update the MAP solution if this one is better.
        # MAP: Maximum A Posteriori solution
        # compute a posteriori solution as: rho(y)*sigma(y,d)
        #
        if ((lpcandidate + llcandidate) > lMAP):
            #
            # accept candidate as MAP
            #
            lMAP = lpcandidate + llcandidate
            mMAP = candidate


        #
        # Record the result.
        #
        mout[k,:] = current
        accrate = nacc / niter


    return mout, mMAP, accrate