import copy
import numpy as np
import sklearn.mixture
from scipy.stats import multivariate_normal

from lib import gmm_generation
import open3d as o3d


def simple_pc_gmm_merge(pc, gmm, min_prob = 1e-4, min_sample_density = []):
    # the approach is simply to merge a point cloud with a gmm
    # assuming we have a nice registration


    points = np.asarray(pc.points)
    num_points = len(points)

    #Approach:
    # assign points
    pc_membership = gmm.gmm_generator.predict(points)
    pc_membership_prob = gmm.gmm_generator.predict_proba(points)

    # get likelihood for each point
    member_range = range(1,gmm.num_gaussians)
    point_member_list = np.empty((1,3))

    for i in member_range:
        associated_points = points[pc_membership == i, :]
        #print("number of associated_points: ", np.shape(associated_points))

        likelihoods = multivariate_normal.pdf(associated_points, mean = gmm.means[i, :], cov=gmm.covariances[i, :, :])
        #print("likelihoods: ", likelihoods)
        # get "single axis likelihood"

        # remove points with p < min_prob
        if isinstance(likelihoods, np.floating):
            continue
        reduced_points = associated_points[likelihoods > min_prob, :]
        #print("reduced point shape: ", np.shape(reduced_points))

        if len(reduced_points) > 0:
            point_member_list = np.append(point_member_list, reduced_points, axis = 0) #.append(reduced_points)

    return_pc = o3d.geometry.PointCloud()
    return_pc.points = o3d.utility.Vector3dVector(point_member_list)

    # sample points when there are too few according to gmm dist!

    '''
    print("memberships: ", pc_membership[1:10])
    print("probabilities: ", pc_membership_prob[1:5,1:10])
    print("pc_membership_prob:  ", (np.sum(pc_membership_prob[1,:])))
    print("merge finished")
    '''
    return return_pc
