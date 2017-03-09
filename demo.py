import numpy as np
import PsiMarginal

ntrials = 300  # number of trials
mu = np.linspace(2, 4, 41)  # threshold/bias grid
sigma = np.linspace(0.05, 1, 21)  # slope grid
x = np.linspace(1.5, 4.5, 61)  # possible stimuli to use
lapse = np.linspace(0, 0.1, 15)  # lapse grid
guessRate = 0.5  # fixed guess rate

# parameters used to simulate observer
muGen = 3
sigmaGen = 0.2
lapseGen = 0.05

thresholdPrior = ('normal', 3, 2)  # truncated normal distribution as prior
slopePrior = ('gamma', 2, 0.3)  # truncated gamma distribution as prior
lapsePrior = ('beta', 2, 20)  # truncated beta distribution as prior

# initialize algorithm
psi = PsiMarginal.Psi(x, Pfunction='cGauss', nTrials=ntrials, threshold=mu, thresholdPrior=thresholdPrior,
                      slope=sigma, slopePrior=slopePrior, guessRate=guessRate, guessPrior=('uniform', None),
                      lapseRate=lapse, lapsePrior=lapsePrior, marginalize=True)

# parameters to generate first response
generativeParams = np.array(([muGen, sigmaGen, guessRate, lapseGen, psi.xCurrent])).T
# [1,4] appose to [0,4] is required for likelihood function, so add an additional dim.
generativeParams = np.expand_dims(generativeParams, 0)

print 'Simulating an observer with mu=%.2f, sigma=%.2f and lapse=%.2f.' % (muGen, sigmaGen, lapseGen)
for i in range(0, ntrials):  # run for length of trials
    r = PsiMarginal.GenerateData(generativeParams, psyfun='cGauss')  # generate simulated response
    psi.addData(r)  # update Psi with response
    print 'Trial %d of %d' % (i, ntrials)
    while psi.xCurrent == None:  # wait until next stimuli is calculated
        pass

    generativeParams[0, 4] = psi.xCurrent  # set new stimuli to present
print 'Estimated parameters of this observer are mu=%.2f, sigma=%.2f and lapse=%.2f.' % (psi.eThreshold,
                                                                                         psi.eSlope,
                                                                                         psi.eLapse)
psi.plot(muRef=muGen, sigmaRef=sigmaGen, lapseRef=lapseGen, guessRef=guessRate)
