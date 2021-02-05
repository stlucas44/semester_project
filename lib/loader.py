import matplotlib.pyplot as plt
import copy
import numpy as np
import open3d as o3d
import scipy
import trimesh
from lib import visualization



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
                              angular_resolution = 1.0):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()

    view_point_index = np.random.randint(0, len(mesh.triangles))
    view_point_dist = np.random.uniform(low = altitude_above_ground[0],
                                        high = altitude_above_ground[1])

    vertice_indexes = np.asarray(mesh.triangles[view_point_index])
    mesh_point = np.asarray(mesh.vertices[vertice_indexes[0]])

    relative_pos = np.asarray(mesh.triangle_normals[view_point_index])
    mesh_center = np.asarray(mesh.vertices).mean(axis = 0)

    pos = relative_pos + mesh_point
    (x, y, z) = pos - mesh_center

    print("mesh center shape: ", mesh_center.shape)

    yaw = np.rad2deg(np.arctan2(-y,-x))
    pitch = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    roll = 0.0

    rpy = (roll, pitch, yaw)

    sensor_max_range = 2 * view_point_dist

    view_point_mesh = view_point_crop(mesh, pos, rpy, sensor_max_range,
                                      sensor_fov, angular_resolution)
    print(type(view_point_mesh))
    return view_point_mesh[0]

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
        #return mesh
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

    plot = True
    if plot:
        print(len(mesh.triangles))
        ax = visualization.visualize_mesh(old_mesh)

        step = 10
        ax.scatter(ray_centers[0,0], ray_centers[0,1], ray_centers[0,2], s = 10, c = 'r')
        ax.scatter((ray_centers + rays)[::step,0], (ray_centers + rays)[::step,1],
                   (ray_centers + rays)[::step,2], s = 1, c = 'g')
        plt.show()
    #TODO (stlucas): keep removed triangles in different mesh
    return mesh, occluded_mesh

def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(int(n_points))
