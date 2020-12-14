import copy
import numpy as np
import sklearn.mixture
import scipy
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from lib import gmm_generation
import open3d as o3d

def view_point_crop(mesh, pos, quat):
    # cut all occupied triangles
    radius = 5
    mesh.hidden_point_removal(pos, radius)

    mpl_visualize(mesh)
    pass


def simple_pc_gmm_merge(pc, gmm, min_prob = 1e-3, min_sample_density = []):
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
    point_member_list_structured = []

    for i in member_range:
        associated_points = points[pc_membership == i, :]
        #print("number of associated_points: ", np.shape(associated_points))

        likelihoods = multivariate_normal.pdf(associated_points, mean = gmm.means[i, :], cov=gmm.covariances[i, :, :])
        #print("likelihoods: ", likelihoods)
        # get "single axis likelihood"

        # remove points with p < min_prob
        if isinstance(likelihoods, np.floating):
            continue
            #likelihoods = np.asarray(likelihoods)
            #print(np.shape(likelihoods))
        reduced_points = associated_points[likelihoods > min_prob, :]
        #print("reduced point shape: ", np.shape(reduced_points))

        if len(reduced_points) > 0:
            point_member_list = np.append(point_member_list, reduced_points, axis = 0) #.append(reduced_points)
        point_member_list_structured.append(reduced_points)

    #analyze_result(pc, gmm, point_member_list_structured)

    print("number of points to be sampled: ", num_points - len(point_member_list))
    sampled_points = gmm.sample_from_gmm(num_points - len(point_member_list))
    print("sampled points shape: ", sampled_points)

    #point_member_list = np.append(point_member_list, sampled_points, axis = 0)

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


def gmm_merge(mesh_gmm, pc_gmm, min_overlap = 1.0):
    '''
    presteps:
    remove occluded gmms from prior (but keep them somewhere!)
    idea:
    1. determine case: overlap, only prior, only measurement
    2. if only prior -> remove from collection
       if only measurement -> keep, add to mixture
       if intersecting
        evaluate if should be kept or not?
    3. add "occluded patterns from prior"
    4. create mesh from gmm. (with outer rim from prior)
    --> leads to perfectly integrated mesh

    evaluation metric:
    watching covariances (setting minimum default cov for mesh, get probabilistic mesh?)
    --> large prior cov, small (flat) measurement covs -> keep measurement
    --> small prior cov, large (bumpy) measurement covs -> keep prior
    --> both equal
    '''


    pass

def stitch_pc():
    # implement something nice!
    pass


def analyze_result(pointcloud, gmm, point_groups):
    print("Gmm weight sum", np.sum(gmm.weights))

    #fig = plt.figure()
    group_lengths = []

    for group in point_groups:
        group_lengths.append(len(group))

    print(len(group_lengths))

    gmm_count = np.size(gmm.weights)
    print(gmm_count)
    plt.subplot(211)
    plt.bar(np.arange(0,gmm_count), gmm.weights)
    plt.subplot(212)
    plt.bar(np.arange(0,len(group_lengths)), group_lengths)
    plt.show()



    # plot weights
    #plot point numbers
    #plot point numbers filtered?

    pass
