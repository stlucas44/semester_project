import matplotlib.pyplot as plt
import copy
from datetime import datetime

import numpy as np
import open3d as o3d
import scipy

import trimesh
import pymesh

from lib import gmm_generation, visualization



def load_measurement(path, scale = 1.0):
    pc = o3d.io.read_point_cloud(path)
    return scale_o3d_object(pc, scale)

def load_mesh(path, scale = 1.0):
    mesh = o3d.io.read_triangle_mesh(path)
    return scale_o3d_object(mesh, scale)

def load_unit_mesh(type = 'flat'):
    mesh = o3d.geometry.TriangleMesh()

    if type == 'flat':
        vertices = np.asarray([[0.0, 0.0, 0.0],
                               [10.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]])
        triangles = [[0.0,1.0,2.0]]

    elif type == "3d":
        vertices = np.asarray([[1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0]])
        triangles = [[0.0,1.0,2.0]]

    elif type == "2trisA":
        vertices = np.asarray([[1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, -1.0, 0.0]])
        triangles = [[0.0, 1.0, 2.0],
                     [0.0, 1.0, 3.0]]
    elif type == "2trisB":
        vertices = np.asarray([[1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [1.0, 1.0, 0.0]])
        triangles = [[0.0, 1.0, 2.0],
                     [0.0, 2.0, 3.0]]


    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def scale_o3d_object(object, scale, scaling_center = np.zeros((3,1))):
    scaling_center = np.zeros((3,1))
    return object.scale(scale, scaling_center)

def automated_view_point_mesh(path, altitude_above_ground = (1.0, 3.0),
                              sensor_fov = [180.0, 180.0],
                              sensor_max_range = 100.0,
                              angular_resolution = 1.0,
                              plot = False):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()

    view_point_index = np.random.randint(0, len(mesh.triangles))
    view_point_dist = np.random.uniform(low = altitude_above_ground[0],
                                        high = altitude_above_ground[1])

    vertice_indexes = np.asarray(mesh.triangles[view_point_index])
    mesh_point = np.asarray(mesh.vertices[vertice_indexes[0]])

    relative_pos = view_point_dist * np.asarray(mesh.triangle_normals[view_point_index])
    mesh_center = np.asarray(mesh.vertices).mean(axis = 0)

    pos = relative_pos + mesh_point
    (x, y, z) = mesh_center - pos

    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    roll = 0.0

    rpy = (roll, pitch, yaw)

    sensor_max_range = 3.0 * view_point_dist

    view_point_mesh, occluded_mesh = view_point_crop_by_trace(
                                      mesh, pos, rpy, sensor_max_range,
                                      sensor_fov, angular_resolution,
                                      plot = plot)
    return view_point_mesh

def view_point_crop_by_cast(mesh, pos, rpy,
                    sensor_max_range = 100.0,
                    sensor_fov = [180.0, 180.0],
                    angular_resolution = 1.0,
                    get_pc = False,
                    plot = False):
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

    old_mesh = copy.deepcopy(mesh)
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
        return None, None
    triangle_index = np.unique(triangle_index)

    #create mask from intersected triangles
    triangle_range = np.arange(len(mesh.triangles))
    mask = [element not in triangle_index for element in triangle_range]
    print("  number of hit triangles: ", len(triangle_index),
          " of ", len(triangle_range))

    anti_mask = [not element for element in mask]
    #remove the nonvisible parts of the mesh
    occluded_mesh = copy.deepcopy(mesh)

    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    occluded_mesh.remove_triangles_by_mask(anti_mask)
    occluded_mesh.remove_unreferenced_vertices()

    if plot:
        ax = visualization.visualize_mesh(old_mesh)
        step = 10
        ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
        ax.scatter((ray_centers + rays)[::step,0], (ray_centers + rays)[::step,1],
                   (ray_centers + rays)[::step,2], s = 1, c = 'g')
        plt.show()
    #TODO (stlucas): keep removed triangles in different mesh
    return mesh, occluded_mesh


def view_point_crop_by_trace(mesh, pos, rpy,
                    sensor_max_range = 100.0,
                    sensor_fov = [180.0, 180.0],
                    angular_resolution = 1.0,
                    get_pc = False,
                    plot = False):
    # cut all occluded triangles
    '''
    Cut all occluded triangles
    Overview:
    1. Transform to trimesh for better handling
    2. Draw rays from each triangle centroid to the camera
    3. Check if angles are in range of the view point
    3. Check if rays intersect other triangles
    4. Apply mask to keep the local mesh

    returns the occlusion mesh
    '''

    #compte classic normals
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    py_mesh = pymesh.meshio.form_mesh(np.asarray(mesh.vertices),
                                   np.asarray(mesh.triangles))

    centroids, a = gmm_generation.get_centroids(py_mesh)
    occluded_mesh = copy.deepcopy(mesh)
    old_mesh = copy.deepcopy(mesh)

    # create rays and trace them ray.ray_pyembree--> pyembree for more speed
    rays = np.asarray(centroids - pos) #(pos - centroids)

    # roll, pitch, yaw
    rpy = np.deg2rad(rpy)
    sensor_fov = np.deg2rad(sensor_fov) / 2.0

    vp_bounds_pitch = (rpy[1] - sensor_fov[1], rpy[1] + sensor_fov[1])
    vp_bounds_yaw = (rpy[2] - sensor_fov[0], rpy[2] + sensor_fov[0])

    in_vp_mask = np.zeros((len(rays),), dtype = bool)

    #remove all points not in viewpoint
    for i, ray in zip(np.arange(0,len(rays)), rays):
        rpy_ray = [0.0, np.arctan2(ray[2], np.sqrt(np.square(ray[1]) + np.square(ray[0]))),
                   np.arctan2(ray[1],ray[0])]


        in_yaw = vp_bounds_yaw[0] <= rpy_ray[2] <= vp_bounds_yaw[1]
        in_pitch = vp_bounds_pitch[0] <= rpy_ray[1] <= vp_bounds_pitch[1]

        in_vp_mask[i] = in_yaw & in_pitch

    mesh.remove_triangles_by_mask([not element for element in in_vp_mask])
    mesh.remove_unreferenced_vertices()

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    if in_vp_mask.sum == 0:
        print("no points in view point")
        return None, occluded_mesh

    local_mesh = trimesh.base.Trimesh(vertices = mesh.vertices,
                                      faces = mesh.triangles,
                                      face_normals = mesh.triangle_normals,
                                      vertex_normals = mesh.vertex_normals)
    centroids = centroids[in_vp_mask]
    rays = rays[in_vp_mask]
    print("  ", len(centroids), " of ", len(in_vp_mask), " in VP")

    local_pos = np.reshape(pos, (1,3))
    multipos = np.repeat(local_pos, len(centroids), axis = 0)

    print(" starting ray trace")
    raytracer = trimesh.ray.ray_triangle.RayMeshIntersector(local_mesh)
    mesh_intersections_mask = raytracer.intersects_any(centroids, -0.99 * rays)

    print("hit : ", np.asarray(mesh_intersections_mask).sum(), " of ",
          mesh_intersections_mask.shape[0])
    mesh.remove_triangles_by_mask(mesh_intersections_mask)
    mesh.remove_unreferenced_vertices()

    anti_mask = [not element for element in in_vp_mask]
    #anti_mask[in_vp_mask] = mesh_intersections_mask

    print("  number of hit triangles: ", np.asarray(mesh.triangles).shape[0],
          " of ", len(anti_mask))

    occluded_mesh.remove_triangles_by_mask(anti_mask)
    occluded_mesh.remove_unreferenced_vertices()

    # edge rpy:

    rpy_b0 = (0.0, vp_bounds_pitch[0], vp_bounds_yaw[0])
    rpy_b1 = (0.0, vp_bounds_pitch[1], vp_bounds_yaw[0])
    rpy_b2 = (0.0, vp_bounds_pitch[0], vp_bounds_yaw[1])
    rpy_b3 = (0.0, vp_bounds_pitch[1], vp_bounds_yaw[1])

    rpy_b = [rpy_b0, rpy_b1, rpy_b2, rpy_b3]
    # vecs =
    length = 3.0

    if plot:
        #ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
        ax = visualization.visualize_mesh(mesh)
        #ax = visualization.visualize_mesh(old_mesh, ax = ax)
        step = 100
        ax.scatter(pos[0], pos[1], pos[2], s = 10, c = 'r')

        ax.scatter((pos + rays)[::step,0], (pos + rays)[::step,1],
                   (pos + rays)[::step,2], s = 1, c = 'g')

        ax.scatter(centroids[::step, 0], centroids[::step, 1], centroids[::step, 2],
                   c = 'c')
        for ray in rays:
            ax.plot([pos[0], pos[0] + ray[0]],
                    [pos[1], pos[1] + ray[1]],
                    [pos[2], pos[2] + ray[2]])
        for element in rpy_b:
            r = length * np.asarray([np.cos(element[2]), np.sin(element[2]), np.sin(element[1])])
            ax.plot([pos[0], pos[0] + r[0]],
                    [pos[1], pos[1] + r[1]],
                    [pos[2], pos[2] + r[2]])
        plt.show()
    #TODO (stlucas): keep removed triangles in different mesh
    return mesh, occluded_mesh

def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(int(n_points))


def get_name(name):
    start = name.rfind("/") + 1
    end = name.rfind(".")
    return name[start:end]

def get_figure_path(params, plt_type):
    folder = "imgs/"
    name = get_name(params['path'])
    date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    date_time = date_time[:date_time.rfind(":")]
    date_time = date_time[(date_time.find("-")+1):]
    return folder + name + "_" + plt_type + "_" + date_time + ".png"
