import numpy as np
import scipy
from scipy.stats import multivariate_normal

import sklearn.mixture

from lib.gmm_generation import Gmm
from lib.visualization import *
from lib import merge

def main():
    # steps for evaluation
    # generate two or more gmms with only few gaussians
    run1()


    # now we run this with different scales of n?

    print("works")
    pass

def run1():
    #visualize overlaying distribution and
    p_means = np.asarray([0.0]).reshape((-1,1))
    p_covs= np.asarray([2.0]).reshape((-1,1))
    prior = Gmm(means = [0.0], covariances = [2.0])
    prior.num_gaussians = 1
    m_means = np.arange(-10, 10).reshape((-1,1))
    m_covs = np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    vis_update(prior, measurement)
    plt.legend()
    plt.show()

    bla = merge.gmm_merge(prior, measurement)




def vis_update(prior, measurement):
    plt.subplot(211)
    vis_gmm(prior, label = "prior")
    plt.subplot(212)
    vis_gmm(measurement, label = "measurement")

def vis_gmm(gmm, label = ""):
    range = np.arange(0, len(gmm.means))
    color=np.random.rand(3,)
    for i in range:
        x = np.linspace(gmm.means[i] - 3 * np.sqrt(gmm.covariances[i]),
                        gmm.means[i] + 3 * np.sqrt(gmm.covariances[i]))
        y = multivariate_normal.pdf(x, mean = gmm.means[i],
                                    cov=gmm.covariances[i])

        plt.plot(x,y, c = color, label = label)

if __name__ == "__main__":
    main()
