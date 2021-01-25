import numpy as np
import open3d as o3d

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
                               [1.0, 0.0, 0.0],
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
                               [0.0, -1.0, 0.0]])
        triangles = [[0.0, 1.0, 2.0],
                     [0.0, 1.0, 3.0]]


    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def scale_o3d_object(object, scale, scaling_center = np.zeros((3,1))):
    scaling_center = np.zeros((3,1))
    return object.scale(scale, scaling_center)

def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(int(n_points))
