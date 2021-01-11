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
    p_covs= np.asarray([5.0]).reshape((-1,1))
    prior = Gmm(means = [0.0], covariances = [2.0])
    prior.num_gaussians = 1
    m_means = np.arange(-10, 10, 2).reshape((-1,1))
    m_covs = np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)


    result, t = merge.gmm_merge(prior, measurement)
    print(result)
    vis_update(prior, measurement, result)
    plt.legend()
    plt.show()

def run2():
    #visualize overlaying distribution and
    p_means = np.asarray([0.0]).reshape((-1,1))
    p_covs= np.asarray([2.0]).reshape((-1,1))
    prior = Gmm(means = [0.0], covariances = [2.0])
    prior.num_gaussians = 1
    m_means = np.zeros(m_means.shape)
    m_covs = np.arange(-10, 10).reshape((-1,1))

    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    vis_update(prior, measurement)
    plt.legend()
    plt.show()

    results, t = merge.gmm_merge(prior, measurement)

def vis_update(prior, measurement, result = None):
    if result is not None:
        fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex = True)
        vis_result(measurement.means, result, ax[2])
        ax[2].set_title('Result')
    else:
        fig, ax = plt.subplots(2, 1, constrained_layout=True)


    ax[0].set_title('Prior')
    vis_gmm(prior, ax[0], label = "prior")

    ax[1].set_title('Measurement')
    colors = np.random.rand(3,measurement.num_gaussians)
    vis_gmm(measurement, ax[1], label = "measurement", color =colors)

def vis_gmm(gmm, ax, label = "", color= None):
    range = np.arange(0, len(gmm.means))
    if color is None:
        color = np.random.rand(3,gmm.num_gaussians)
    for i in range:
        x = np.linspace(gmm.means[i] - 3 * np.sqrt(gmm.covariances[i]),
                        gmm.means[i] + 3 * np.sqrt(gmm.covariances[i]))
        y = multivariate_normal.pdf(x, mean = gmm.means[i],
                                    cov=gmm.covariances[i])

        #ax.plot(x,y, c = color, label = label)
        ax.plot(x,y, c = color[:,i])

def vis_result(means, results, ax, colors = None):
    if colors is None:
        color = np.random.rand(3,len(means))

    ax.scatter(means, results, c = colors)

if __name__ == "__main__":
    main()
