
'''
P(C)  --> P(S|C) --> P(W|S,R)
      --> P(R|C) -->

What is the solution to the following?
P(R=1 | S=1, W=1)
'''

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
# ...

### Estimate P(R=1 | S=1, W=1) through Gibbs sampling ###
# ...



