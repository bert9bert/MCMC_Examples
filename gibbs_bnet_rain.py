
'''
PROBLEM SETUP: Suppose that we have a Bayesian network set up as follows.

C=1 if cloudy, =0 otherwise
S=1 if sprinkers were just on, =0 otherwise
R=1 if just rained, =0 otherwise
W=1 if grass is wet, =0 otherwise

P(C)  --> P(S|C) --> P(W|S,R)
      --> P(R|C) -->

Suppose that the network's nodes' are Bernoulli random variables.

Furthermore, suppose that we have already estimated the parameters for the
network's nodes' random variables using some dataset (or that we were given
the parameters).

QUESTION: What is the marginal probability of it just rained given that
the sprinklers were just on and that the grass is wet?
i.e. What is P(R=1 | S=1, W=1) ?

THIS SCRIPT estimates this probability using both a brute force sampling
method and using a Gibbs sampler.
'''


import numpy as np

### Program parameters ###
NUM_BRUTEFORCE_SAMPLES = 100000

NUM_GIBBS_SAMPLES = 100000
NUM_GIBBS_BURNIN = 1000


### Define prob mass functions of random variables at the Bayesian network's nodes ###

# P(C)
def pmf_cloudy(is_cloudy):
    if is_cloudy:
        return 0.50
    else:
        return 0.50

# P(S|C)
def pmf_sprinker_given_cloudy(is_sprinkler, is_cloudy):
    if is_cloudy:
        if is_sprinkler:
            return 0.10
        else:
            return 0.90
    else:
        if is_sprinkler:
            return 0.50
        else:
            return 0.50

# P(R|C)
def pmf_rain_given_cloudy(is_rain, is_cloudy):
    if is_cloudy:
        if is_rain:
            return 0.80
        else:
            return 0.20
    else:
        if is_rain:
            return 0.20
        else:
            return 0.80

# P(W|S,R)
def pmf_wetgrass_given_sprinker_rain(is_wetgrass, is_sprinkler, is_rain):
    if is_sprinkler:
        if is_rain:
            if is_wetgrass:
                return 0.99
            else:
                return 0.01
        else:
            if is_wetgrass:
                return 0.90
            else:
                return 0.10
    else:
        if is_rain:
            if is_wetgrass:
                return 0.90
            else:
                return 0.10
        else:
            if is_wetgrass:
                return 0.01
            else:
                return 0.99

### Define conditional probabilities ###
def pmf_cloudy_given_rest(is_cloudy, is_rain, is_sprinkler, is_wetgrass):
    if is_sprinkler and is_wetgrass:
        numer = pmf_cloudy(is_cloudy)*pmf_sprinker_given_cloudy(True, is_cloudy)*pmf_rain_given_cloudy(is_rain, is_cloudy)
        denom = pmf_cloudy(True)*pmf_sprinker_given_cloudy(True, True)*pmf_rain_given_cloudy(is_rain, True) + \
            pmf_cloudy(False)*pmf_sprinker_given_cloudy(True, False)*pmf_rain_given_cloudy(is_rain, False)
        return numer/denom
    else:
        raise Exception('This part of the PMF has not been programmed in since it not necessary for this example.')

def pmf_rain_given_rest(is_rain, is_cloudy, is_sprinkler, is_wetgrass):
    if is_sprinkler and is_wetgrass:
        numer = pmf_rain_given_cloudy(is_rain, is_cloudy)*pmf_wetgrass_given_sprinker_rain(True, True, is_rain)
        denom = pmf_rain_given_cloudy(True, is_cloudy)*pmf_wetgrass_given_sprinker_rain(True, True, True) + \
            pmf_rain_given_cloudy(False, is_cloudy)*pmf_wetgrass_given_sprinker_rain(True, True, False)
        return numer/denom
    else:
        raise Exception('This part of the PMF has not been programmed in since it not necessary for this example.')


### Estimate P(R=1 | S=1, W=1) through brute force sampling ###
bf_C = np.random.binomial(1, pmf_cloudy(True), NUM_BRUTEFORCE_SAMPLES)
bf_S = np.array([ np.random.binomial(1, pmf_sprinker_given_cloudy(True, bf_C[i]), 1)[0] for i in range(NUM_BRUTEFORCE_SAMPLES) ])
bf_R = np.array([ np.random.binomial(1, pmf_rain_given_cloudy(True, bf_C[i]), 1)[0] for i in range(NUM_BRUTEFORCE_SAMPLES) ])
bf_W = np.array([ np.random.binomial(1, pmf_wetgrass_given_sprinker_rain(True, bf_S[i], bf_R[i]), 1)[0] for i in range(NUM_BRUTEFORCE_SAMPLES) ])

bruteforce_condprob_rain = sum(bf_R[i] and bf_S[i] and bf_W[i] for i in range(NUM_BRUTEFORCE_SAMPLES)) / sum(bf_S[i] and bf_W[i] for i in range(NUM_BRUTEFORCE_SAMPLES))


### Estimate P(R=1 | S=1, W=1) through Gibbs sampling ###
## Use Gibbs sampler to sample P(R, C | S=1, W=1)

is_rain_init = True
is_cloudy_init = True

x_list = [(is_rain_init, is_cloudy_init)]

for g in range(NUM_GIBBS_BURNIN + NUM_GIBBS_SAMPLES):
    # sample rain conditional on last value of cloudy
    is_rain_this = np.random.binomial(1, pmf_rain_given_rest(True, x_list[-1][1], True, True), 1)[0]
    # sample cloudy given this sample of rain
    is_cloudy_this = np.random.binomial(1, pmf_cloudy_given_rest(True, is_rain_this, True, True), 1)[0]
    # store samples
    x_list.append((is_rain_this, is_cloudy_this))

## remove burn-in samples and store
x_mat = np.array(x_list[-NUM_GIBBS_SAMPLES:])

## compute conditional marginal
gibbs_condprob_rain = sum(x_mat[:,0])/NUM_GIBBS_SAMPLES

### Display Results ###
print('Brute Force Estimate of    P(R=1 | S=1, W=1) = %6.4f' % bruteforce_condprob_rain)
print('Gibbs Sampling Estimate of P(R=1 | S=1, W=1) = %6.4f' % gibbs_condprob_rain)
