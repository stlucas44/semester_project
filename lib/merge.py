import copy
import numpy as np
import sklearn.mixture
import scipy
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from lib import gmm_generation, visualization
import open3d as o3d
import trimesh

from numba import jit, cuda

def view_point_crop(mesh, pos, rpy,
                    sensor_max_range = 100.0,
                    sensor_fov = [180.0, 180.0],
                    angular_resolution = 1.0,
                    get_pc = False):
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
    print(" starting ray trace")
    triangle_index = raytracer.intersects_first(ray_centers, rays)
    triangle_index = np.delete(triangle_index, np.where(triangle_index == -1))
    if triangle_index.size == 0:
        print(" rays do not intersect! mesh outside sensor range")
        return mesh
    triangle_index = np.unique(triangle_index)

    #create mask from intersected triangles
    triangle_range = np.arange(len(mesh.triangles))
    mask = [element not in triangle_index for element in triangle_range]
    print("number of hit triangles: ", len(triangle_index),
          " of ", len(triangle_range))

    anti_mask = [not element for element in mask]
    #remove the nonvisible parts of the mesh
    occluded_mesh = copy.deepcopy(mesh)

    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    occluded_mesh.remove_triangles_by_mask(anti_mask)
    occluded_mesh.remove_unreferenced_vertices()

    plot = False
    if plot:
        ax = visualization.visualize_mesh(mesh)
        step = 10
        ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
        ax.scatter((ray_centers + rays)[::step,0], (ray_centers + rays)[::step,1],
                   (ray_centers + rays)[::step,2], s = 1, c = 'g')
        plt.show()
    #TODO (stlucas): keep removed triangles in different mesh
    return mesh, occluded_mesh


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

#@jit(target ="cuda")
def gmm_merge(prior_gmm, measurement_gmm, p_crit = 0.95, sample_size = 100,
              sample_ratio = 1.0, exit_early = False):
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
    #print("prior range:", prior_range, "measurement_range", measurement_range)

    #create match matrix to check for the cases
    match = np.zeros((len(prior_range),len(measurement_range)), dtype=bool)
    score = np.zeros((len(prior_range),len(measurement_range)))

    for i in prior_range:
        if( i % 20 == 0):
            print("  intersection computing: ", i, " of ", len(prior_range))
        mean = prior_gmm.means[i]
        cov = prior_gmm.covariances[i]

        #mask = get_intersection_type_simple(measurement_gmm.means, mean = mean, cov = cov, min_likelihood = 0.1)
        p_value = np.zeros((measurement_gmm.num_gaussians,))
        result = np.zeros((measurement_gmm.num_gaussians,),dtype=bool)

        for j in measurement_range:
            local_result, p_value[j] = get_intersection_type_hotelling(mean, cov,
                                    measurement_gmm.means[j],
                                    measurement_gmm.covariances[j],
                                    p_crit = p_crit,
                                    sample_size = sample_size,
                                    sample_ratio = sample_ratio)
            result[j] = local_result
            match[i,j] = local_result
            score[i,j] = p_value[j]
        plot = False
        if plot:
            plt.plot(measurement_range, t)
            plt.show()
        #print(p_value)
        if exit_early:
            return result, p_value

        #print("mask created")

    plot_sums = True
    if plot_sums:
        visualization.visualize_match_matrix(match, score)

    prior_mask, measurement_mask, matched_mixture_tuples = create_masks_simple(
        match, prior_gmm, measurement_gmm)

    prior_mask, measurement_mask = create_masks(match, score)

    final_mixture = gmm_generation.merge_gmms(measurement_gmm, measurement_mask,
                                              prior_gmm, prior_mask)
    print(measurement_mask, prior_mask)
    final_mixture_tuple = (measurement_gmm.extract_gmm(measurement_mask),
                           prior_gmm.extract_gmm(prior_mask))

    # find all measurement_gmms that overlap with prior_gmms
    resampled_mixture = resample_mixture(final_mixture_tuple)
    measurement_only_mixture = measurement_gmm.extract_gmm(np.invert(measurement_mask))

    # create new (all true) masks and merge
    resampled_mixture_mask = np.ones((len(resampled_mixture.means),), dtype=bool)
    measurement_only_mixture_mask = np.ones((len(measurement_only_mixture.means),), dtype=bool)
    final_mixture =gmm_generation.merge_gmms(
            resampled_mixture, resampled_mixture_mask,
            measurement_only_mixture, measurement_only_mixture_mask)

    return matched_mixture_tuples, final_mixture, final_mixture_tuple

def get_intersection_type_simple(points, mean, cov, min_likelihood = 0.1):
    appearence_likelihood = multivariate_normal.pdf(points, mean = mean, cov = cov)

    return appearence_likelihood > min_likelihood

def get_intersection_type_hotelling(meanA, covA, meanB, covB, p_crit = 0.95, sample_size = 100.0, sample_ratio = 1.0):
    #source: https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_statistic
    if isinstance(meanA, float):
        meanA = np.array([meanA]).reshape((1,1))
    if isinstance(meanB, float):
        meanB = np.array([meanB]).reshape((1,1))


    meanA = meanA.reshape((-1,1))
    meanB = meanB.reshape((-1,1))

    # TODO(stlucas): should we adapt sample sizes according what

    # mixture weights, covariance size...
    nA = sample_size
    nB =sample_ratio * sample_size

    t_squared = get_t_squared(meanA, covA, nA, meanB, covB, nB)
    #t_squared = get_t_squared2(meanA, covA, nA, meanB, covB, nB)
    v = get_v(covA, nA, covB, nB)
    #print("t² = ", t_squared, "t = ", np.sqrt(t_squared))

    # trying to fix hotelling:
    p = v # sometimes assigned as p, v is the degree of freedom!
    # found approximation of degrees of freedom as this: https://en.wikipedia.org/wiki/Behrens%E2%80%93Fisher_problem
    #print("v = ", v)
    #p = 6
    # trying alternate
    statistic = t_squared * (nA + nB-p-1)/(p*(nA + nB-2))

    F = scipy.stats.f(p, nA + nB - p - 1)

    p_value = 1 - F.cdf(statistic)


    return p_value > p_crit, p_value

def get_t_squared(meanA, covA, nA, meanB, covB, nB):
    # this is similar to https://www.r-bloggers.com/2020/10/hotellings-t2-in-julia-python-and-r/
    S = ((nA - 1.0) * covA + (nB - 1.0) * covB) / (nA + nB - 2.0)
    if S.shape == (1,):
        S = S.reshape((1,1))

    t_squared = (nA * nB) * (nA + nB) * np.linalg.multi_dot([(meanA - meanB).T,np.linalg.inv(S),(meanA - meanB)])
    return t_squared

def get_t_squared2(y_1, S_1, n_1, y_2, S_2, n_2):
    if isinstance(S_1, float):
        S_1 = np.reshape(S_1, (1,1))
        S_2 = np.reshape(S_2, (1,1))

    return (y_1 - y_2).T.dot(np.linalg.inv(S_1/n_1 + S_2/n_2)).dot(y_1 - y_2)

def get_v(S_1, n_1, S_2, n_2):
    Sn1 = S_1/n_1
    Sn2 = S_2/n_2

    if isinstance(S_1, float):
        Sn1 = np.reshape(Sn1, (1,1))
        Sn2 = np.reshape(Sn2, (1,1))

    A = ((Sn1+Sn2).dot(Sn1+Sn2)).trace()
    B =  (Sn1+Sn2).trace()**2
    C = ((Sn1.dot(Sn1)).trace() + Sn1.trace() **2)/(n_1-1)
    D = ((Sn2.dot(Sn2)).trace() + Sn2.trace() **2)/(n_2-1)

    return (A+B)/(C+D)

def decide_on_mixtures(gmm0, gmm1, rule = "cov"):
    if rule == "cov":
        #for now, go only for
        alpha = 0
        beta = 1
        if cost(gmm1, alpha, beta) > cost(gmm0, alpha, beta):
            return 0
        else:
            return 1

    elif rule =="mesh_first":
        return 1

def cost(gmm, alpha, beta):
    cov_measure = [min(np.linalg.eigvals(gmm.covariances[:]))]
    print("Cov measure = ", cov_measure)
    cost = alpha * len(gmm.means) + beta * np.max(cov_measure)
    print("cost: ", cost)

    return cost


def create_masks_simple(match, prior_gmm, measurement_gmm):
    #TODO(stlucas): create new gmms with these mappings from match
    vert_sums = np.sum(match, axis = 0) #column sum
    hor_sums = np.sum(match, axis = 1) # row sum

    merged_mixtures = list()
    matched_mixture_tuples = list()

    # find all prior gmms that overlap with measurements
    prior_mask = np.ones((len(prior_gmm.means),), dtype=bool)
    measurement_mask = np.ones((len(measurement_gmm.means),), dtype=bool)

    iterator = 0
    for score in hor_sums:
        if score == 0:
            prior_mask[iterator] = False
            iterator = iterator + 1
            continue
        local_prior_mask = np.zeros(hor_sums.shape, dtype = bool)
        local_prior_mask[iterator] = True
        local_measurement_mask = np.asarray([match[iterator,:] > 0]).reshape(-1,)
        merged_gmm = gmm_generation.merge_gmms(measurement_gmm, local_measurement_mask,
                                               prior_gmm, local_prior_mask)
        merged_mixtures.append(merged_gmm)
        mixture_tuple = (measurement_gmm.extract_gmm(local_measurement_mask),
                         prior_gmm.extract_gmm(local_prior_mask))
        matched_mixture_tuples.append(mixture_tuple)
        iterator = iterator + 1

        #fill final merge mask:
        result = decide_on_mixtures(*mixture_tuple, rule='mesh_first') # returs 0 for first
        measurement_mask[local_measurement_mask] = not result # for now we set the current matches to zero
        prior_mask[local_prior_mask] = result

    return prior_mask, measurement_mask, matched_mixture_tuples

def create_masks(match, score):
    column_sums = np.sum(match, axis = 0) #column sum
    row_sums = np.sum(match, axis = 1) # row sum

    measurement_mask = column_sums != 0
    prior_mask = row_sums != 0

    return prior_mask, measurement_mask

def resample_mixture(gmm_tuple):
    # implement something nice!
    weights = [0.5, 0.5]
    n = int(1e5)
    point_collection = list()

    pc = o3d.geometry.PointCloud()
    for (gmm, weights) in zip(gmm_tuple, weights):
        if gmm.num_gaussians == 0:
            print("no mixtures, continue!")
            continue

        local_pc = gmm.sample_from_gmm(n)
        point_collection.append(np.asarray(local_pc.points))
        #print("point_collection shape: ", point_collection.shape)

    point_collection = np.asarray(point_collection).reshape(-1,3)
    print("point_collection shape: ", point_collection.shape)
    pc.points = o3d.utility.Vector3dVector(point_collection)

    resampled_gmm = gmm_generation.Gmm(weights = [], means = [], covariances = [])
    #NOTE: when not initialzing to zero -> somehow there are means around!

    #resampled_gmm.pc_simple_gmm(pc)
    resampled_gmm.pc_hgmm(pc)

    return resampled_gmm

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

    pass
