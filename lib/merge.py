import copy
import numpy as np
import sklearn.mixture
import scipy
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from lib import gmm_generation, visualization
import open3d as o3d
import trimesh

#from tf.transformations import quaternion_from_euler

def view_point_crop(mesh, pos, rpy, sensor_max_range = 100.0, sensor_fov = [180.0, 180.0], angular_resolution = 1.0):
    # cut all occluded triangles

    #compte classic normals
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()


    #transform to trimesh
    #trimesh.ray.ray_pyembree.RayMeshIntersector --> pyembree for more speed
    local_mesh = trimesh.base.Trimesh(vertices = mesh.vertices,
                                      faces = mesh.triangles,
                                      face_normals = mesh.triangle_normals,
                                      vertex_normals = mesh.vertex_normals)

    # create rays and trace them
    raytracer = trimesh.ray.ray_triangle.RayMeshIntersector(local_mesh)
    sensor_u = np.arange(-sensor_fov[0]/2, sensor_fov[0]/2, angular_resolution)
    sensor_v = np.arange(-sensor_fov[1]/2, sensor_fov[1]/2, angular_resolution)
    sensor_u, sensor_v = np.meshgrid(sensor_u, sensor_v)

    #assuming a front_left_up frame for the camera: --> watch out, its circular!
    rays = sensor_max_range * np.array([np.cos(sensor_u) * np.cos(sensor_v),
                               np.sin(sensor_u)* np.cos(sensor_v),
                               np.sin(sensor_v)])\
                               .reshape((sensor_u.shape[0]*sensor_v.shape[1], 3))

    ray_centers = np.tile(pos, (rays.shape[0], 1))
    print(ray_centers[1:10, :])

    #transform to global frame!
    r = scipy.spatial.transform.Rotation.from_euler('zyx', rpy[::-1], degrees=True)
    rays = r.apply(rays)


    #check intersected triangles
    print("starting ray trace")
    triangle_index = raytracer.intersects_first(ray_centers, rays)
    print("triangle index shapes: ", len(triangle_index), "\n first ten objects:",
        triangle_index[:10], "object type: ", type(triangle_index))
    triangle_index = np.delete(triangle_index, np.where(triangle_index == -1))
    triangle_index = np.unique(triangle_index)

    #these are dummy matches!
    triangle_index = np.asarray(range(1,100))

    print("number of hit triangles: ", len(triangle_index))
    print("number of triangles: ", len(mesh.triangles))
    triangle_range = np.arange(len(mesh.triangles))
    print("number of triangle range: ", len(triangle_range))
    print(triangle_range[0:10], triangle_range[-10:-1])
    #mask = [True for i in range(0,len(mesh.triangles)) if i in triangle_index else False]
    mask = [element not in triangle_index for element in triangle_range]
    print("mask length", len(mask))
    #frame: assuming the world frame is ENU -> yaw increases from east to north

    #remove the nonvisible parts of the mesh
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    ax = visualization.visualize_mesh(mesh)
    step = 10
    ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
    #ax.scatter((ray_centers + rays)[::step,0], (ray_centers + rays)[::step,1],
    #           (ray_centers + rays)[::step,2], s = 1, c = 'g')

    plt.show()

    return []


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
    Assumptions:
    * Sensor sees all objects that are not occluded in its fov
    * Localization is proper!
    * Rough alignment alignment is guaranteed


    pre-steps:
    remove occluded gmms from prior (but keep them somewhere!)

    algo:
    1. determine case: overlap, only prior, only measurement
    2. if only prior -> remove from collection
       if only measurement -> keep, add to mixture
       if intersecting
        evaluate if should be kept or not?
    3. add "occluded patterns from prior"
    4. create mesh from gmm. (with outer rim from prior)
    --> leads to perfectly integrated mesh

    evaluation metric: (aka regularized covariances?)
    watching covariances (setting minimum default cov for mesh, get probabilistic mesh?)
    --> large prior cov, small (flat) measurement covs -> keep measurement
    --> small prior cov, large (bumpy) measurement covs -> keep prior
    --> both equal

    cost = alpha * n_gmm + beta * 1/cov_measure

    cov_measure = mean_cov, max_cov, weighted mean_cov (weight large ones heavier?)?

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
