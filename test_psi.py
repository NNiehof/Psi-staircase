import PsiMarginal
from sklearn.utils.extmath import cartesian
import numpy as np


def test_cgauss_likelihood():
    mu = np.array([0], dtype='float')
    sigma = np.array([2], dtype='float')
    x = np.linspace(-1, 2, 2)
    lapse = np.array([0], dtype='float')
    parameters = cartesian((mu, sigma, lapse, x))
    proportionMethod = PsiMarginal.pf(parameters, psyfun='cGauss')
    samples = np.random.normal(mu, sigma, (200000, 1))
    proportionSamples = np.empty([2, ])
    proportionSamples[0] = np.mean(samples <= x[0])  # cdf is p(X<=x), compute this through sampling to check likelihood
    proportionSamples[1] = np.mean(samples <= x[1])
    np.testing.assert_almost_equal(proportionSamples, proportionMethod, decimal=2) == 1


def test_generate_data():
    mu = np.array([0], dtype='float')
    sigma = np.array([2], dtype='float')
    x = np.array([1], dtype='float')
    lapse = np.array([0], dtype='float')
    parameters = np.array(([mu, sigma, lapse, x])).T
    proportion = PsiMarginal.pf(parameters, psyfun='cGauss')
    samples = PsiMarginal.GenerateData(parameters, psyfun='cGauss', ntrials=100000)
    meanSamples = np.mean(samples)
    varSamples = np.var(samples)
    np.testing.assert_almost_equal(meanSamples, proportion, decimal=2) == 1
    np.testing.assert_almost_equal(varSamples, proportion * (1 - proportion), decimal=2) == 1


def test_gamma_equal_lambda():
    mu = np.linspace(-100, 100, 2)
    sigma = np.linspace(2, 200, 2)
    x = np.linspace(-200, 200, 3)
    lapse = np.linspace(0, 0.1, 4)
    guess = lapse
    psi = PsiMarginal.Psi(x, Pfunction='cGauss', nTrials=50, threshold=mu, thresholdPrior=('uniform', None),
                          slope=sigma, slopePrior=('uniform', None),
                          guessRate=guess, guessPrior=('uniform', None), lapseRate=lapse, lapsePrior=('uniform', None),
                          marginalize=True)
    assert psi.gammaEQlambda == True
    guess = np.array([0.5], dtype='float')
    psi2 = PsiMarginal.Psi(x, Pfunction='cGauss', nTrials=50, threshold=mu, thresholdPrior=('uniform', None),
                           slope=sigma, slopePrior=('uniform', None),
                           guessRate=guess, guessPrior=('uniform', None), lapseRate=lapse, lapsePrior=('uniform', None),
                           marginalize=True)
    assert psi2.gammaEQlambda == False
