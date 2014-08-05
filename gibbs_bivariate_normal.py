'''
Use Gibbs sampling to draw from a bivariate Normal distribution.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

### Program parameters ###
MAXITER = 100
N = 100
BURNIN = 100

### Parameters to draw with ###
mu0 = 10.0
std0 = 15.0

mu1 = 30.0
std1 = 25.0

rho = 0.75

### set initial values ###
sample_init = np.array([3, 4])

### Perform Gibbs sampling ###

# set up list to hold samples, including the initial value
gibbs_samples = [ sample_init ]

for itr in range(BURNIN + N):
    sample_last = gibbs_samples[-1]

    x0sample = np.random.normal(mu0 + std0/std1*rho*(sample_last[1]-mu1),
        np.sqrt((1-rho**2)*std0**2),
        1)[0]

    x1sample = np.random.normal(mu1 + std1/std0*rho*(x0sample-mu0),
        np.sqrt((1-rho**2)*std1**2),
        1)[0]

    gibbs_samples.append(np.array([x0sample, x1sample]))

samples = np.array(gibbs_samples[-N:])



### Display results ###
# draw scatter plot of drawn data
plt.scatter(samples[:,0], samples[:,1])

# overlay contours of bivariate normal drawn from
delta = 0.25
x = np.arange(-50.0, 100.0, delta)
y = np.arange(-50.0, 100.0, delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, std0, std1, mu0, mu1, rho*std0*std1) * 100.0

CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)

# label chart
plt.title('Draws from a Bivariate Normal by Gibbs Sampling \n' + r'$\mu_0=%5.2f, \mu_1=%5.2f, \sigma_0=%5.2f, \sigma_1=%5.2f, \rho=%4.2f$' % (mu0, mu1, std0, std1, rho))
plt.xlabel('Component 0')
plt.ylabel('Component 1')

plt.show()
