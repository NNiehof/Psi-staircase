# -*- coding: utf-8 -*-
"""
Nynke Niehof (2016) Radboud University Nijmegen; Donders Institute for Brain, Cognition and Behaviour.

Psi adaptive staircase procedure for use in psychophysics, as described in Kontsevich & Tyler (1999)
and psi-marginal staircase as described in Prins(2013). Implementation based on the Palamedes toolbox for Matlab.

References:

Kontsevich, L. L. & Tyler, C. W. (1999). Bayesian adaptive estimation of psychometric slope and threshold.
    Vision Research, 39, 2729-2737.
Prins, N & Kingdom, F. A. A. (2009). Palamedes: Matlab routines for analyzing psychophysical data.
    http://www.palamedestoolbox.org 
Prins, N. (2013). The psi-marginal adaptive method: How to give nuisance parameters the attention they
    deserve (no more, no less). Journal of Vision, 13(7):3, 1-17.
"""

import numpy as np
from sklearn.utils.extmath import cartesian
from scipy.stats import norm
from scipy.special import erfc
import threading


def PF(parameters, psyfun='cGauss'):
    """Generate conditional probabilities from psychometric function.
    
    Arguments
    ---------
        parameters: ndarray containing parameters as columns
            alpha   : threshold
            
            beta    : slope
            
            gamma   : guessing rate (optional), default is 0.5
            
            lambda  : lapse rate (optional), default is 0.04
            
            x       : stimulus intensity
            
        psyfun (str): type of psychometric function.
            'cGauss' cumulative Gaussian
            
            'Gumbel' Gumbel, aka log Weibull

    Returns
    -------
    1D-array of conditional probabilities p(response | alpha,beta,gamma,lambda,x)
    """

    ## Unpack parameters
    if np.size(parameters,1) == 5:
        [alpha,beta,gamma,llambda,x] = np.transpose(parameters)
    elif np.size(parameters,1) == 4:
        [alpha,beta,llambda,x] = np.transpose(parameters)
        gamma           = llambda
    elif np.size(parameters,1) == 3:
        [alpha,beta,x]  = np.transpose(parameters)
        gamma           = 0.5
        llambda         = 0.04
    else: # insufficient number of parameters will give a flat line
        psyfun          = None
        gamma           = 0.5
        llambda         = 0.04
    
    ## Psychometric function
    ones                = np.ones(np.shape(alpha))
    if psyfun == 'cGauss':
        #F(x; alpha, beta) = Normcdf(alpha, beta) = 1/2 * erfc(-beta * (x-alpha) /sqrt(2))
        pf              = ones/2* erfc(np.multiply(-beta, (np.subtract(x, alpha))) /np.sqrt(2))
    elif psyfun == 'Gumbel':
        # F(x; alpha, beta) = 1 - exp(-10^(beta(x-alpha)))
        pf              = ones - np.exp(-np.power((np.multiply(ones,10.0)), (np.multiply(beta, (np.subtract(x, alpha))))))
    else:
        # flat line if no psychometric function is specified
        pf              = np.ones(np.shape(alpha))
    y = gamma + np.multiply((ones - gamma - llambda), pf)
    return y


class Psi:
    """Find the stimulus intensity with minimum expected entropy for each trial, to determine the psychometric function.
  
    Psi adaptive staircase procedure for use in psychophysics.   
    
    Arguments
    ---------
        stimRange :
            range of possible stimulus intensities.

        Pfunction (str) : type of psychometric function to use.
            'cGauss' cumulative Gaussian
            
            'Gumbel' Gumbel, aka log Weibull
        
        nTrials :
            number of trials
            
        threshold :
            (alpha) range of possible threshold values to search
            
        thresholdPrior (tuple) : type of prior probability distribution to use.
            Also: slopePrior, guessPrior, lapsePrior.
            
            ('normal',0,1): normal distribution, mean and standard deviation.
            
            ('uniform',None) : uniform distribution, mean and standard deviation not defined.
            
        slope :
            (beta) range of possible slope values to search
        
        slopePrior :
            see thresholdPrior
            
        guessRate :
            (gamma) range of possible guessing rate values to search
        
        guessPrior :
            see thresholdPrior
        
        lapseRate :
            (lambda) range of possible lapse rate values to search
        
        lapsePrior :
            see thresholdPrior
        
        marginalize (bool) :
            If True, marginalize out the lapse rate and guessing rate before finding the stimulus
            intensity of lowest expected entropy. This uses the Prins (2013) method to include the guessing and lapse rate
            into the probability disctribution. These rates are then marginalized out, and only the threshold and slope are included
            in selection of the stimulus intensity.
            
            If False, lapse rate and guess rate are included in the selection of stimulus intensity.
    
    How to use
    ----------
        Create a psi object instance with all relevant arguments. Selecting a correct search space for the threshold,
        slope, guessing rate and lapse rate is important for the psi procedure to function well. If an estimate for
        one of the parameters ends up at its (upper or lower) limit, the result is not reliable, and the procedure
        should be repeated with a larger search range for that parameter.
        
        Example:
            >>> s   = range(-5,5) # possible stimulus intensities
            obj = Psi(s)
        
        The stimulus intensity to be used in the current trial can be found in the class variable xCurrent.
        
        Example:
            >>> stim = obj.xCurrent
        
        After each trial, update the psi staircase with the subject response, by calling the addData method.
        
        Example:
            >>> obj.addData(resp)
    
    Note: if package sklearn (part of scikit) is not available, use list(itertools.product()) instead of cartesian()
    """
    
    def __init__(self, stimRange, Pfunction='cGauss', nTrials=50, threshold=None, thresholdPrior=('uniform',None), slope=None, slopePrior=('uniform',None),
                 guessRate=None, guessPrior=('uniform',None), lapseRate=None, lapsePrior=('uniform',None), marginalize=True):
        
        ## Psychometric function parameters
        self.stimRange      = stimRange # range of stimulus intensities
        self.threshold      = np.arange(-10,10,0.05)
        self.slope          = np.arange(0.005,100,0.05)
        self.guessRate      = np.arange(0.0,0.11,0.1)
        self.lapseRate      = np.arange(0.0,0.11,0.1)
        self.marginalize    = marginalize # marginalize out nuisance parameters gamma and lambda?
            
        if threshold is not None:
            self.threshold  = threshold
        if slope is not None:
            self.slope      = slope
        if guessRate is not None:
            self.guessRate  = guessRate
        if lapseRate is not None:
            self.lapseRate  = lapseRate

        # remove any singleton dimensions
        self.threshold      = np.squeeze(self.threshold)
        self.slope          = np.squeeze(self.slope)
        self.guessRate      = np.squeeze(self.guessRate)
        self.lapseRate      = np.squeeze(self.lapseRate)

        ## Priors        
        self.priorAlpha     = self.__genprior(self.threshold, *thresholdPrior)
        self.priorBeta      = self.__genprior(self.slope, *slopePrior)
        self.priorGamma     = self.__genprior(self.guessRate, *guessPrior)
        self.priorLambda    = self.__genprior(self.lapseRate, *lapsePrior)
        
        # if guess rate equals lapse rate, and they have equal priors,
        # then gamma can be left out, as the distributions will be the same
        self.gammaEQlambda       = all([[all(self.guessRate == self.lapseRate)], [all(self.priorGamma == self.priorLambda)]])
        
        # likelihood: table of conditional probabilities p(response | alpha,beta,gamma,lambda,x)
        # prior: prior probability over all parameters p_0(alpha,beta,gamma,lambda)
        if self.gammaEQlambda:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.lapseRate), len(self.stimRange))
            self.parameters = cartesian((self.threshold, self.slope, self.lapseRate, self.stimRange))   
            self.likelihood = PF(self.parameters, psyfun=Pfunction)
            self.likelihood = np.reshape(self.likelihood, self.dimensions) # dims: (alpha, beta, lambda, x)
            self.pr         = cartesian((self.priorAlpha, self.priorBeta, self.priorLambda))
            self.prior      = np.prod(self.pr, axis=1) # row-wise products of prior probabilities
            self.prior      = np.reshape(self.prior, self.dimensions[:-1]) # dims: (alpha, beta, lambda)
        else:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.guessRate), len(self.lapseRate, len(self.stimRange)))
            self.parameters = cartesian((self.threshold, self.slope, self.guessRate, self.lapseRate, self.stimRange))   
            self.likelihood = PF(self.parameters, psyfun=Pfunction)
            self.likelihood = np.reshape(self.likelihood, self.dimensions) # dims: (alpha, beta, gamma, lambda, x)
            self.pr         = cartesian((self.priorAlpha, self.priorBeta, self.priorGamma, self.priorLambda))
            self.prior      = np.prod(self.pr, axis=1) # row-wise products of prior probabilities
            self.prior      = np.reshape(self.prior, self.dimensions[:-1]) # dims: (alpha, beta, gamma, lambda)
        
        # normalize prior
        self.prior          = self.prior / np.sum(self.prior)

        ## Set probability density function to prior
        self.pdf            = np.copy(self.prior)

        # settings
        self.iTrial         = 0
        self.nTrials        = nTrials
        self.stop           = 0
        self.response       = []

        ## Generate the first stimulus intensity
        self.minEntropyStim()


    def __genprior(self, x, distr='uniform', mu=0, sig=1): # prior probability distribution
        if distr == 'uniform':
            nx              = len(x)
            p               = np.ones(nx)/nx
        elif distr == 'normal':
            p               = norm.pdf(x, mu, sig)
        else:
            nx              = len(x)
            p               = np.ones(nx)/nx
        return p


    def __entropy(self, pdf): # Shannon entropy of probability density function
        ## Marginalize out all nuisance parameters, i.e. all except alpha and beta
        postDims            = np.ndim(pdf)
        if self.marginalize == True:
            while postDims  > 3: # marginalize out second-to-last dimension, last dim is x
                pdf         = np.sum(pdf, axis=-2)
                postDims    -= 1
        # find expected entropy, suppress divide-by-zero and invalid value warnings
        # as this is handled by the NaN redefinition to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy             = np.multiply(pdf, np.log(pdf))
        entropy[np.isnan(entropy)] = 0 # define 0*log(0) to equal 0
        dimSum              = tuple(range(postDims-1)) # dimensions to sum over. also a Chinese dish
        entropy             = -(np.sum(entropy, axis=dimSum))
        return entropy
        
         
    def minEntropyStim(self):
        """Find the stimulus intensity based on the expected information gain.
        
        Minimum Shannon entropy is used as selection criterion for the stimulus intensity in the upcoming trial.
        """
        self.pdf            = self.pdf
        self.nX             = len(self.stimRange)
        self.nDims          = np.ndim(self.pdf)

        # make pdf the same dims as conditional prob table likelihood
        self.pdfND          = np.expand_dims(self.pdf, axis=self.nDims) # append new axis
        self.pdfND          = np.tile(self.pdfND, (self.nX)) # tile along new axis
        
        ## Probabilities of response r (succes, failure) after presenting a stimulus
        ## with stimulus intensity x at the next trial, multiplied with the prior (pdfND)
        self.pTplus1success = np.multiply(self.likelihood, self.pdfND)
        self.pTplus1failure = self.pdfND - self.pTplus1success
        
        ## Probability of success or failure given stimulus intensity x, p(r|x)
        self.sumAxes        = tuple(range(self.nDims)) # sum over all axes except the stimulus intensity axis
        self.pSuccessGivenx = np.sum(self.pTplus1success, axis=self.sumAxes)
        self.pFailureGivenx = np.sum(self.pTplus1failure, axis=self.sumAxes)
        
        ## Posterior probability of parameter values given stimulus intensity x and response r
        ## p(alpha, beta | x, r)
        self.posteriorTplus1success = self.pTplus1success / self.pSuccessGivenx
        self.posteriorTplus1failure = self.pTplus1failure / self.pFailureGivenx
        
        ## Expected entropy for the next trial at intensity x, producing response r
        self.entropySuccess = self.__entropy(self.posteriorTplus1success)
        self.entropyFailure = self.__entropy(self.posteriorTplus1failure)      
        self.expectEntropy  = np.multiply(self.entropySuccess, self.pSuccessGivenx) + np.multiply(self.entropyFailure, self.pFailureGivenx)
        self.minEntropyInd  = np.argmin(self.expectEntropy) # index of smallest expected entropy
        self.xCurrent       = self.stimRange[self.minEntropyInd] # stim intensity at minimum expected entropy
        
        self.iTrial         += 1
        if self.iTrial == (self.nTrials -1):
            self.stop = 1


    def addData(self, response):
        """
        Add the most recent response to start calculating the next stimulus intensity
        
        Arguments
        ---------
            Response:
                1: correct/right
                
                0: incorrect/left
        """
        self.response.append(response)
        
        self.xCurrent   = None        
        
        ## Keep the posterior probability distribution that corresponds to the recorded response
        if response ==1:
            # select the posterior that corresponds to the stimulus intensity of lowest entropy
            self.pdf    = self.posteriorTplus1success[Ellipsis, self.minEntropyInd]
        elif response == 0:
            self.pdf    = self.posteriorTplus1failure[Ellipsis, self.minEntropyInd]

        # normalize the pdf
        self.pdf        = self.pdf / np.sum(self.pdf)

        ## Marginalized probabilities per parameter      
        if self.gammaEQlambda:
            self.pThreshold = np.multiply(self.threshold,   np.sum(self.pdf, axis=(1,2)))
            self.pSlope     = np.multiply(self.slope,       np.sum(self.pdf, axis=(0,2)))
            self.pLapse     = np.multiply(self.lapseRate,   np.sum(self.pdf, axis=(0,1)))
            self.pGuess     = self.pLapse
        else:
            self.pThreshold = np.multiply(self.threshold,   np.sum(self.pdf, axis=(1,2,3)))
            self.pSlope     = np.multiply(self.slope,       np.sum(self.pdf, axis=(0,2,3)))
            self.pLapse     = np.multiply(self.lapseRate,   np.sum(self.pdf, axis=(0,1,2)))
            self.pGuess     = np.multiply(self.guessRate,   np.sum(self.pdf, axis=(0,1,3)))
        ## Expected values of parameters
        self.eThreshold     = np.sum(self.pThreshold)
        self.eSlope         = np.sum(self.pSlope)
        self.eLapse         = np.sum(self.pLapse)
        self.eGuess         = np.sum(self.pGuess)
        
        ## Start calculating the next minimum entropy stimulus
        threading.Thread(target=self.minEntropyStim).start()

