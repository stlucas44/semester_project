import copy
import numpy as np
import sklearn.mixture
import scipy
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from lib import gmm_generation, visualization
import open3d as o3d
import trimesh

def view_point_crop(mesh, pos, rpy, sensor_max_range = 100.0, sensor_fov = [180.0, 180.0], angular_resolution = 1.0):
    # cut all occluded triangles
    '''
    Cut all occluded triangles
    Overview:
    1. Transform to trimesh for better handling
    2. Create the sensor model based on range, fov and angular resolution
        --> we assume a front-left-up frame for the sensor fov center
    3. transform the model to localized pose
    3. Raycast for each pixel and check which of them intersect first.
    4. Apply mask to keep the local mesh

    returns the occlusion mesh
    '''

    #compte classic normals
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    #transform to trimesh
    local_mesh = trimesh.base.Trimesh(vertices = mesh.vertices,
                                      faces = mesh.triangles,
                                      face_normals = mesh.triangle_normals,
                                      vertex_normals = mesh.vertex_normals)

    # create rays and trace them ray.ray_pyembree--> pyembree for more speed
    raytracer = trimesh.ray.ray_triangle.RayMeshIntersector(local_mesh)
    alpha = np.arange(-sensor_fov[0]/2, sensor_fov[0]/2, angular_resolution)
    beta = np.arange(-sensor_fov[1]/2, sensor_fov[1]/2, angular_resolution)
    alpha, beta = np.meshgrid(alpha, beta)

    #assuming a front_left_up frame for the camera
    rays = sensor_max_range * np.array([np.cos(alpha/180.0 * np.pi),
                               np.sin(alpha/180.0 * np.pi),
                               np.sin(beta/180.0 * np.pi)])\
                               .reshape((3, alpha.shape[0]*beta.shape[1]))
    rays = rays.T
    ray_centers = np.tile(pos, (rays.shape[0], 1))

    #transform to global frame!
    r = scipy.spatial.transform.Rotation.from_euler('ZYX', rpy[::-1], degrees=True)
    rays = r.apply(rays)

    #check intersected triangles and remove dublicates
    print("starting ray trace")
    triangle_index = raytracer.intersects_first(ray_centers, rays)
    triangle_index = np.delete(triangle_index, np.where(triangle_index == -1))
    if triangle_index.size == 0:
        print("rays do not intersect! mesh outside sensor range")
        return mesh
    triangle_index = np.unique(triangle_index)

    #create mask from intersected triangles
    triangle_range = np.arange(len(mesh.triangles))
    mask = [element not in triangle_index for element in triangle_range]
    print("number of hit triangles: ", len(triangle_index))


    #remove the nonvisible parts of the mesh
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    plot = False
    if plot:
        ax = visualization.visualize_mesh(mesh)
        step = 10
        ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
        ax.scatter((ray_centers + rays)[::step,0], (ray_centers + rays)[::step,1],
                   (ray_centers + rays)[::step,2], s = 1, c = 'g')
        plt.show()
    #TODO (stlucas): keep removed triangles in different mesh
    return mesh #, removed_mesh


def simple_pc_gmm_merge(pc, gmm, min_prob = 1e-3, min_sample_density = []):
    '''
    merging a point cloud with
    project a pointcloud into a GMM.
    points with likelihood lower than the bound are removed
    missing points are going to be resampled from the distribution

    returns pointcloud
    '''
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


def gmm_merge(prior_gmm, measurement_gmm, min_overlap = 1.0):
    '''
    Assumptions:
    * Sensor sees all objects that are not occluded in its fov
    * Localization is proper!
    * Rough alignment alignment is guaranteed


    pre-steps:
    remove occluded gmms from prior (but keep them somewhere!) --> view_point_crop

    algo:
    1. determine case: overlap, only prior, only measurement [0,1,2]
        * t-test
        * mean_test near enough
    2. if only prior -> remove from collection
       if only measurement -> keep, add to mixture
       if intersecting
        evaluate if should be kept or not?
    3. add "occluded patterns from prior"
    4. create mesh from gmm. (with outer rim from prior?)
    --> leads to perfectly integrated mesh

    evaluation metric: (aka regularized covariances?)
    watching covariances (setting minimum default cov for mesh, get probabilistic mesh?)
    --> large prior cov, small (flat) measurement covs -> keep measurement
    --> small prior cov, large (bumpy) measurement covs -> keep prior
    --> both equal

    cost = alpha * n_gmm + beta * 1/cov_measure

    cov_measure = mean_cov, max_cov, weighted mean_cov (weight large ones heavier?)?

    '''
    # input: (occluded) mesh_means, mesh_covs
    #        pc_means, pc_covs
    prior_range = np.arange(0, prior_gmm.num_gaussians)
    measurement_range = np.arange(0, measurement_gmm.num_gaussians)
    keep = np.zeros(len(prior_range,), dtype=bool)
    for i in prior_range:
        mean = prior_gmm.means[i]
        cov = prior_gmm.covariances[i]

        #mask = get_intersection_type_simple(measurement_gmm.means, mean = mean, cov = cov, min_likelihood = 0.1)
        t = np.zeros((measurement_gmm.num_gaussians,))
        for j in measurement_range:

            t[j] = get_intersection_type(mean, cov,
                                        measurement_gmm.means[j],
                                        measurement_gmm.covariances[j])
        plot = True
        if plot:
            plt.plot(measurement_range, t)
            plt.show()
        #print(min(t))

        #print("mask created")

    pass

def get_intersection_type_simple(points, mean, cov, min_likelihood = 0.1):
    appearence_likelihood = multivariate_normal.pdf(points, mean = mean, cov = cov)

    return appearence_likelihood > min_likelihood

def get_intersection_type(meanA, covA, meanB, covB):
    #source: https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_statistic
    meanA = meanA.reshape((-1,1))
    meanB = meanB.reshape((-1,1))

    sample_size = 10.0
    n_x = 1.0 * sample_size
    n_y = 1.0 * sample_size

    S = ((n_x - 1.0) * covA + (n_y - 1.0) * covB) / (n_x + n_y - 2.0)

    t_squared = (n_x * n_y) * (n_x + n_y) * np.linalg.multi_dot([(meanA - meanB).T,np.linalg.inv(S),(meanA - meanB)])
    #print("t² = ", t_squared, "t = ", np.sqrt(t_squared))

    return np.sqrt(t_squared)

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
