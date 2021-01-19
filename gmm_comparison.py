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
    #run1()

    # now we run this with different scales of n?
    #run2()

    # try different sample_sizes
    run3()
    # try different ratios
    #run4()

    pass

def run1(): # vary means
    #visualize overlaying distribution and
    prior = Gmm(means = [0.0], covariances = [5.0])
    prior.num_gaussians = 1
    m_means = np.arange(0, 20, 1).reshape((-1,1))
    m_covs = 5.0 * np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    result, p_value = merge.gmm_merge(prior, measurement, p_crit = 0.95)

    print(p_value)
    vis_update(prior, measurement, p_value)
    #vis_update(prior, measurement, [not elem for elem in result],
    #           path="imgs/1dMerge.png")
    plt.legend()
    plt.show()

def run2(): # vary variances
    #visualize overlaying distribution and
    prior = Gmm(means = [0.0], covariances = [2.0])
    prior.num_gaussians = 1

    m_covs = np.arange(0.1, 20, 2).reshape((-1,1))
    m_means = 3 * np.ones(len(m_covs))

    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    result, t = merge.gmm_merge(prior, measurement)
    vis_update(prior, measurement, t)
    #vis_update(prior, measurement, [not elem for elem in result],
    #           path="imgs/1dMerge.png")
    plt.legend()
    plt.show()

def run3(): # vary sample sizes (keeping covs as in run 2)
    prior = Gmm(means = [0.0], covariances = [5.0])
    prior.num_gaussians = 1
    m_means = np.arange(0, 20, 2).reshape((-1,1))
    m_covs = 5.0 * np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    max_log_space = 5
    sample_sizes = 2 * np.logspace(0,max_log_space, num = max_log_space + 1)
    ax = vis_update(prior, measurement)
    for sample in sample_sizes:
        print("sample_size = ", sample)
        result, p_value = merge.gmm_merge(prior, measurement, p_crit = 0.95, sample_size = sample)
        ax[2].plot(measurement.means, p_value, label = "sample_size = " + str(sample))

        print(result)
        print(p_value)
    plt.legend()
    plt.show()

def run4(): # vary sample sizes (keeping covs as in run 2)
    prior = Gmm(means = [0.0], covariances = [5.0])
    prior.num_gaussians = 1
    m_means = np.arange(0, 20, 1).reshape((-1,1))
    m_covs = 5.0 * np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)


    sample_sizes = np.linspace(1.0,10.0, num = 10)
    for sample in sample_sizes:
        print("sample_size = ", 1/sample)
        result, t = merge.gmm_merge(prior, measurement, sample_size = 1000, sample_ratio = 1/sample)
        plt.plot(measurement.means, p_value, label = "sample_size = " + str(1/sample))
        print(result)
    plt.legend()
    plt.show()

def present_merge(): # vary sample sizes (keeping covs as in run 2)
    prior = Gmm(means = [0.0, 5.0], covariances = [5.0, 1.0])
    prior.num_gaussians = 1
    m_means =[-10.0, -1.0, 1.0]
    m_covs = 1.0 * np.ones(m_means.shape)
    measurement = Gmm(means = m_means, covariances = m_covs)
    measurement.num_gaussians = len(m_means)

    result, t = merge.gmm_merge(prior, measurement)
    vis_update(prior, measurement, result, path="imgs/1dMerge.png")
    #vis_update(prior, measurement, [not elem for elem in result],
    #           path="imgs/1dMerge.png")
    plt.legend()
    plt.show()


def vis_update(prior, measurement, result = None, path = None):
    colors = np.random.rand(3,measurement.num_gaussians)

    if result is not None:
        if measurement.means[0] != measurement.means[1]:
            fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex = True)
            vis_result(measurement.means, result, ax[2], colors.T)
        else:
            fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex = True)
            vis_result(measurement.covariances, result, ax[2], colors.T)

        ax[2].set_title('T-test Result')
    else:
        fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex = True)


    ax[1].set_title('Prior')
    vis_gmm(prior, ax[1], label = "Prior")

    ax[0].set_title('Measurement')
    vis_gmm(measurement, ax[0], label = "measurement", color = colors)

    if path is not None:
        plt.savefig(path)

    return ax

def vis_gmm(gmm, ax, label = "", color= None):
    range = np.arange(0, len(gmm.means))
    if color is None:
        color = np.random.rand(3,gmm.num_gaussians)
    for i in range:
        #print(gmm.covariances[i])
        x = np.linspace(gmm.means[i] - 3 * np.sqrt(gmm.covariances[i]),
                        gmm.means[i] + 3 * np.sqrt(gmm.covariances[i]))
        y = multivariate_normal.pdf(x, mean = gmm.means[i],
                                    cov=gmm.covariances[i])
        label = "mean = " + str(gmm.means[i]) + "  cov = " + str(gmm.covariances[i])
        ax.plot(x,y, c = color[:,i], label = label)

def vis_result(means, results, ax, colors = None):
    if colors is None:
        color = np.random.rand(3,len(means))

    ax.scatter(means, results, c = colors)

if __name__ == "__main__":
    main()
