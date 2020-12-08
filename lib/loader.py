import numpy as np
import open3d as o3d

def load_measurement(path, scale = 1.0):
    pc = o3d.io.read_point_cloud(path)
    return scale_o3d_object(pc, scale)

def load_mesh(path, scale = 1.0):
    mesh = o3d.io.read_triangle_mesh(path)
    return scale_o3d_object(mesh, scale)

def scale_o3d_object(object, scale, scaling_center = np.zeros((3,1))):
    scaling_center = np.zeros((3,1))
    return object.scale(scale, scaling_center)
