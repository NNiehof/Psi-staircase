# -*- coding: utf-8 -*-
"""
Copyright © 2016, N. Niehof, Radboud University Nijmegen

PsiMarginal is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PsiMarginal is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PsiMarginal. If not, see <http://www.gnu.org/licenses/>.

---

Psi adaptive staircase procedure for use in psychophysics, as described in Kontsevich & Tyler (1999)
and psi-marginal staircase as described in Prins(2013). Implementation based on the psi-marginal method
in the Palamedes toolbox (version 1.8.1) for Matlab.

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
import scipy
from scipy.stats import norm, beta, gamma
from scipy.special import erfc
import threading
import matplotlib.pyplot as plt


def pf(parameters, psyfun='cGauss'):
    """Generate conditional probabilities from psychometric function.

    Arguments
    ---------
        parameters: ndarray (float64) containing parameters as columns
            mu   : threshold

            sigma    : slope

            gamma   : guessing rate (optional), default is 0.2

            lambda  : lapse rate (optional), default is 0.04

            x       : stimulus intensity

        psyfun  : type of psychometric function.
                'cGauss' cumulative Gaussian

                'Gumbel' Gumbel, aka log Weibull

    Returns
    -------
    1D-array of conditional probabilities p(response | mu,sigma,gamma,lambda,x)
    """

    # Unpack parameters
    if np.size(parameters, 1) == 5:
        [mu, sigma, gamma, llambda, x] = np.transpose(parameters)
    elif np.size(parameters, 1) == 4:
        [mu, sigma, llambda, x] = np.transpose(parameters)
        gamma = llambda
    elif np.size(parameters, 1) == 3:
        [mu, sigma, x] = np.transpose(parameters)
        gamma = 0.2
        llambda = 0.04
    else:  # insufficient number of parameters will give a flat line
        psyfun = None
        gamma = 0.2
        llambda = 0.04
    # Psychometric function
    ones = np.ones(np.shape(mu))
    if psyfun == 'cGauss':
        # F(x; mu, sigma) = Normcdf(mu, sigma) = 1/2 * erfc(-sigma * (x-mu) /sqrt(2))
        z = np.divide(np.subtract(x, mu), sigma)
        p = 0.5 * erfc(-z / np.sqrt(2))
    elif psyfun == 'Gumbel':
        # F(x; mu, sigma) = 1 - exp(-10^(sigma(x-mu)))
        p = ones - np.exp(-np.power((np.multiply(ones, 10.0)), (np.multiply(sigma, (np.subtract(x, mu))))))
    elif psyfun == 'Weibull':
        # F(x; mu, sigma)
        p = 1 - np.exp(-(np.divide(x, mu)) ** sigma)
    else:
        # flat line if no psychometric function is specified
        p = np.ones(np.shape(mu))
    y = gamma + np.multiply((ones - gamma - llambda), p)
    return y


def GenerateData(parameters, psyfun='cGauss', ntrials=None):
    """Generate conditional probabilities from psychometric function.

    Arguments
    ---------
        parameters: [1,4] or [1,5] ndarray (float64) containing parameters as columns
            mu   : threshold

            sigma    : slope

            gamma   : guessing rate (optional), default is 0.2

            lambda  : lapse rate (optional), default is 0.04, if not present we assume lambda = gamma

            x       : stimulus intensity

        psyfun  : type of psychometric function.
                'cGauss' cumulative Gaussian

                'Gumbel' Gumbel, aka log Weibull

        ntrials : number of trials we want to simulate, default is a single scalar

    Returns
    -------
    scalar (ntrials=None) or 1D array of bernoulli variables sampled with probability p(r/mu,sigma,gamma,lambda,x)
    """
    lik = pf(parameters, psyfun=psyfun)
    r = np.random.binomial(1, lik, ntrials)
    return r


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
            (sigma) range of possible slope values to search

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

        The stimulus intensity to be used in the current trial can be found in the field xCurrent.

        Example:
            >>> stim = obj.xCurrent
        NOTE: if obj.xCurrent returns None, the calculation is not yet finished.
        This can be avoided by waiting until xCurrent has a numeric value, e.g.:
            >>> while obj.xCurrent == None:
                    pass # hang in this loop until the psi calculation has finished
                stim = obj.xCurrent

        After each trial, update the psi staircase with the subject response, by calling the addData method.

        Example:
            >>> obj.addData(resp)
    """

    def __init__(self, stimRange, Pfunction='cGauss', nTrials=50, threshold=None, thresholdPrior=('uniform', None),
                 slope=None, slopePrior=('uniform', None),
                 guessRate=None, guessPrior=('uniform', None), lapseRate=None, lapsePrior=('uniform', None),
                 marginalize=True, thread=True):

        # Psychometric function parameters
        self.stimRange = stimRange  # range of stimulus intensities
        self.version = 1.0
        self.threshold = np.arange(-10, 10, 0.1)
        self.slope = np.arange(0.005, 20, 0.1)
        self.guessRate = np.arange(0.0, 0.11, 0.05)
        self.lapseRate = np.arange(0.0, 0.11, 0.05)
        self.marginalize = marginalize  # marginalize out nuisance parameters gamma and lambda?
        self.psyfun = Pfunction
        self.thread = thread

        if threshold is not None:
            self.threshold = threshold
            if np.shape(self.threshold) == ():
                self.threshold = np.expand_dims(self.threshold, 0)
        if slope is not None:
            self.slope = slope
            if np.shape(self.slope) == ():
                self.slope = np.expand_dims(self.slope, 0)
        if guessRate is not None:
            self.guessRate = guessRate
            if np.shape(self.guessRate) == ():
                self.guessRate = np.expand_dims(self.guessRate, 0)
        if lapseRate is not None:
            self.lapseRate = lapseRate
            if np.shape(self.lapseRate) == ():
                self.lapseRate = np.expand_dims(self.lapseRate, 0)

        # Priors
        self.thresholdPrior = thresholdPrior
        self.slopePrior = slopePrior
        self.guessPrior = guessPrior
        self.lapsePrior = lapsePrior

        self.priorMu = self.__genprior(self.threshold, *thresholdPrior)
        self.priorSigma = self.__genprior(self.slope, *slopePrior)
        self.priorGamma = self.__genprior(self.guessRate, *guessPrior)
        self.priorLambda = self.__genprior(self.lapseRate, *lapsePrior)

        # if guess rate equals lapse rate, and they have equal priors,
        # then gamma can be left out, as the distributions will be the same
        self.gammaEQlambda = all((all(self.guessRate == self.lapseRate), all(self.priorGamma == self.priorLambda)))
        # likelihood: table of conditional probabilities p(response | alpha,sigma,gamma,lambda,x)
        # prior: prior probability over all parameters p_0(alpha,sigma,gamma,lambda)
        if self.gammaEQlambda:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.lapseRate), len(self.stimRange))
            self.likelihood = np.reshape(
                pf(cartesian((self.threshold, self.slope, self.lapseRate, self.stimRange)), psyfun=Pfunction), self.dimensions)
            # row-wise products of prior probabilities
            self.prior = np.reshape(
                np.prod(cartesian((self.priorMu, self.priorSigma, self.priorLambda)), axis=1), self.dimensions[:-1])
        else:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.guessRate), len(self.lapseRate), len(self.stimRange))
            self.likelihood = np.reshape(
                pf(cartesian((self.threshold, self.slope, self.guessRate, self.lapseRate, self.stimRange)), psyfun=Pfunction), self.dimensions)
            # row-wise products of prior probabilities
            self.prior = np.reshape(
                np.prod(cartesian((self.priorMu, self.priorSigma, self.priorGamma, self.priorLambda)), axis=1), self.dimensions[:-1])

        # normalize prior
        self.prior = self.prior / np.sum(self.prior)

        # Set probability density function to prior
        self.pdf = np.copy(self.prior)

        # settings
        self.iTrial = 0
        self.nTrials = nTrials
        self.stop = 0
        self.response = []
        self.stim = []

        # Generate the first stimulus intensity
        self.minEntropyStim()

    def __genprior(self, x, distr='uniform', mu=0, sig=1):
        """Generate prior probability distribution for variable.

        Arguments
        ---------
            x   :  1D numpy array (float64)
                    points to evaluate the density at.

            distr :  string
                    Distribution to use a prior :
                        'uniform'   (default) discrete uniform distribution

                        'normal'   normal distribution

                        'gamma'    gamma distribution

                        'beta'     beta distribution

            mu :  scalar float
                first parameter of distr distribution (check scipy for parameterization)

            sig : scalar float
                second parameter of distr distribution

        Returns
        -------
        1D numpy array of prior probabilities (unnormalized)
        """
        if distr == 'uniform':
            nx = len(x)
            p = np.ones(nx) / nx
        elif distr == 'normal':
            p = norm.pdf(x, mu, sig)
        elif distr == 'beta':
            p = beta.pdf(x, mu, sig)
        elif distr == 'gamma':
            p = gamma.pdf(x, mu, scale=sig)
        else:
            nx = len(x)
            p = np.ones(nx) / nx
        return p

    def meta_data(self):
        import time
        import sys
        metadata = {}
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        metadata['date'] = date
        metadata['Version'] = self.version
        metadata['Python Version'] = sys.version
        metadata['Numpy Version'] = np.__version__
        metadata['Scipy Version '] = scipy.__version__
        metadata['psyFunction'] = self.psyfun
        metadata['thresholdGrid'] = self.threshold.tolist()
        metadata['thresholdPrior'] = self.thresholdPrior
        metadata['slopeGrid'] = self.slope.tolist()
        metadata['slopePrior'] = self.slopePrior
        metadata['gammaGrid'] = self.guessRate.tolist()
        metadata['gammaPrior'] = self.guessPrior
        metadata['lapseGrid'] = self.lapseRate.tolist()
        metadata['lapsePrior'] = self.lapsePrior
        return metadata

    def __entropy(self, pdf):
        """Calculate shannon entropy of posterior distribution.
        Arguments
        ---------
            pdf :   ndarray (float64)
                    posterior distribution of psychometric curve parameters for each stimuli


        Returns
        -------
        1D numpy array (float64) : Shannon entropy of posterior for each stimuli
        """
        # Marginalize out all nuisance parameters, i.e. all except alpha and sigma
        postDims = np.ndim(pdf)
        if self.marginalize == True:
            while postDims > 3:  # marginalize out second-to-last dimension, last dim is x
                pdf = np.sum(pdf, axis=-2)
                postDims -= 1
        # find expected entropy, suppress divide-by-zero and invalid value warnings
        # as this is handled by the NaN redefinition to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.multiply(pdf, np.log(pdf))
        entropy[np.isnan(entropy)] = 0  # define 0*log(0) to equal 0
        dimSum = tuple(range(postDims - 1))  # dimensions to sum over. also a Chinese dish
        entropy = -(np.sum(entropy, axis=dimSum))
        return entropy

    def minEntropyStim(self):
        """Find the stimulus intensity based on the expected information gain.

        Minimum Shannon entropy is used as selection criterion for the stimulus intensity in the upcoming trial.
        """
        self.pdf = self.pdf
        self.nX = len(self.stimRange)
        self.nDims = np.ndim(self.pdf)

        # make pdf the same dims as conditional prob table likelihood
        self.pdfND = np.expand_dims(self.pdf, axis=self.nDims)  # append new axis
        self.pdfND = np.tile(self.pdfND, (self.nX))  # tile along new axis

        # Probabilities of response r (succes, failure) after presenting a stimulus
        # with stimulus intensity x at the next trial, multiplied with the prior (pdfND)
        self.pTplus1success = np.multiply(self.likelihood, self.pdfND)
        self.pTplus1failure = self.pdfND - self.pTplus1success

        # Probability of success or failure given stimulus intensity x, p(r|x)
        self.sumAxes = tuple(range(self.nDims))  # sum over all axes except the stimulus intensity axis
        self.pSuccessGivenx = np.sum(self.pTplus1success, axis=self.sumAxes)
        self.pFailureGivenx = np.sum(self.pTplus1failure, axis=self.sumAxes)

        # Posterior probability of parameter values given stimulus intensity x and response r
        # p(alpha, sigma | x, r)
        self.posteriorTplus1success = self.pTplus1success / self.pSuccessGivenx
        self.posteriorTplus1failure = self.pTplus1failure / self.pFailureGivenx

        # Expected entropy for the next trial at intensity x, producing response r
        self.entropySuccess = self.__entropy(self.posteriorTplus1success)
        self.entropyFailure = self.__entropy(self.posteriorTplus1failure)
        self.expectEntropy = np.multiply(self.entropySuccess, self.pSuccessGivenx) + np.multiply(self.entropyFailure,
                                                                                                 self.pFailureGivenx)
        self.minEntropyInd = np.argmin(self.expectEntropy)  # index of smallest expected entropy
        self.xCurrent = self.stimRange[self.minEntropyInd]  # stim intensity at minimum expected entropy

        self.iTrial += 1
        if self.iTrial == (self.nTrials - 1):
            self.stop = 1

    def addData(self, response):
        """
        Add the most recent response to start calculating the next stimulus intensity

        Arguments
        ---------
            response: (int)
                1: correct/right

                0: incorrect/left
        """
        self.stim.append(self.xCurrent)
        self.response.append(response)

        self.xCurrent = None

        # Keep the posterior probability distribution that corresponds to the recorded response
        if response == 1:
            # select the posterior that corresponds to the stimulus intensity of lowest entropy
            self.pdf = self.posteriorTplus1success[Ellipsis, self.minEntropyInd]
        elif response == 0:
            self.pdf = self.posteriorTplus1failure[Ellipsis, self.minEntropyInd]

        # normalize the pdf
        self.pdf = self.pdf / np.sum(self.pdf)

        # Marginalized probabilities per parameter
        if self.gammaEQlambda:
            self.pThreshold = np.sum(self.pdf, axis=(1, 2))
            self.pSlope = np.sum(self.pdf, axis=(0, 2))
            self.pLapse = np.sum(self.pdf, axis=(0, 1))
            self.pGuess = self.pLapse
        else:
            self.pThreshold = np.sum(self.pdf, axis=(1, 2, 3))
            self.pSlope = np.sum(self.pdf, axis=(0, 2, 3))
            self.pLapse = np.sum(self.pdf, axis=(0, 1, 2))
            self.pGuess = np.sum(self.pdf, axis=(0, 1, 3))

        # Distribution means as expected values of parameters
        self.eThreshold = np.sum(np.multiply(self.threshold, self.pThreshold))
        self.eSlope = np.sum(np.multiply(self.slope, self.pSlope))
        self.eLapse = np.sum(np.multiply(self.lapseRate, self.pLapse))
        self.eGuess = np.sum(np.multiply(self.guessRate, self.pGuess))

        # Distribution std of parameters
        self.stdThreshold = np.sqrt(np.sum(np.multiply((self.threshold - self.eThreshold) ** 2, self.pThreshold)))
        self.stdSlope = np.sqrt(np.sum(np.multiply((self.slope - self.eSlope) ** 2, self.pSlope)))
        self.stdLapse = np.sqrt(np.sum(np.multiply((self.lapseRate - self.eLapse) ** 2, self.pLapse)))
        self.stdGuess = np.sqrt(np.sum(np.multiply((self.guessRate - self.eGuess) ** 2, self.pGuess)))

        # Start calculating the next minimum entropy stimulus
        if self.thread:
            threading.Thread(target=self.minEntropyStim).start()
        else:
            self.minEntropyStim()

    def plot(self, muRef=None, sigmaRef=None, lapseRef=None, guessRef=None, save=False):
        """
        Plot marginal distribution of mu, sigma, lapse and posterior distribution of psychometric curve.
        Title of the parameter posteriors indicate the mean +- sd of parameters marginal posterior.

        Arguments
        ---------
            muRef : scalar float
                    Reference value of mu used to generate the psychometric curve.

            sigmaRef: scalar float
                    Reference value of sigma used to generate the psychometric curve.

            lapseRef: scalar float
                    Reference value of lapse rate used to generate the psychometric curve.

            guessRef: scalar float
                    Reference value of lapse rate used to generate the psychometric curve.

            psyfun: string
                    Psychometric function used to generate the data

            save: boolean
                    Flag whether to save figure
                    True : save figure
                    False: don't save figure
        """

        if all((muRef, sigmaRef, lapseRef)):
            ref = True  # reference values exist
            if guessRef:
                nx = len(self.stimRange)
                params = np.array(([np.tile(muRef, nx), np.tile(sigmaRef, nx), np.tile(guessRef, nx),
                                    np.tile(lapseRef, nx), self.stimRange])).T
                curve = pf(params, psyfun=self.psyfun)
            else:  # assume guess rate and lapse are equal
                nx = len(self.stimRange)
                params = np.array(
                    ([np.tile(muRef, nx), np.tile(sigmaRef, nx), np.tile(lapseRef, nx), self.stimRange])).T
                curve = pf(params, psyfun=self.psyfun)
        else:
            ref = False

        if self.gammaEQlambda:
            postmean = np.sum(self.likelihood * self.pdfND, axis=(0, 1, 2))  # mean
            poststd = np.sqrt(
                np.sum(self.likelihood ** 2 * self.pdfND, axis=(0, 1, 2)) - postmean ** 2)  # std
        else:
            postmean = np.sum(self.likelihood * self.pdfND, axis=(0, 1, 2, 3))  # mean
            poststd = np.sqrt(
                np.sum(self.likelihood ** 2 * self.pdfND, axis=(0, 1, 2, 3)) - postmean ** 2)  # std

        plt.figure(figsize=(8, 7))
        plt.subplot(2, 2, 1)
        if ref:
            plt.plot(self.stimRange, curve, 'k', label='True')
        plt.plot(self.stimRange, postmean, 'k--', label='Estimated')
        plt.fill_between(self.stimRange, postmean + poststd, postmean - poststd,
                         alpha=0.2, facecolor='k')
        plt.plot(self.stim, self.response, 'ok', label='Response', markersize=5)
        plt.title('Trial ' + str(self.iTrial - 1))
        plt.legend(loc='upper left', frameon=False, fontsize=10)
        plt.xlabel('x')
        plt.ylabel('p(response)')

        plt.subplot(2, 2, 2)
        plt.plot(self.threshold, self.pThreshold, 'k')
        plt.xlabel(r'$\mu$')
        plt.ylabel('Posterior Probability')
        plt.title('Posterior ' + r'$\mu$=' + str(np.round(self.eThreshold, 3)) +
                  r' $\pm$ ' + str(np.round(self.stdThreshold, 3)))
        plt.axvline(muRef, color='k')
        plt.axvline(self.eThreshold, color='k', linestyle='dashed')

        plt.subplot(2, 2, 3)
        plt.plot(self.lapseRate, self.pLapse, 'k')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Posterior Probability')
        plt.title('Posterior ' + r'$\lambda$=' + str(np.round(self.eLapse, 3)) +
                  r' $\pm$ ' + str(np.round(self.stdLapse, 3)))
        plt.axvline(lapseRef, color='k')
        plt.axvline(self.eLapse, color='k', linestyle='dashed')

        plt.subplot(2, 2, 4)
        plt.plot(self.slope, self.pSlope, 'k')
        plt.xlabel(r'$\sigma$')
        plt.ylabel('Posterior Probability')
        plt.title('Posterior ' + r'$\sigma$=' + str(np.round(self.eSlope, 3)) +
                  r' $\pm$ ' + str(np.round(self.stdSlope, 3)))
        plt.axvline(sigmaRef, color='k')
        plt.axvline(self.eSlope, color='k', linestyle='dashed')
        plt.tight_layout()
        if save:
            plt.savefig('PsiCurve.png')
        plt.show()
