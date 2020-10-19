import numpy as np
import matplotlib

import pyvista as pv
from pyvista import examples

import laspy

point_cloud_file= '../data/mini_spiez/2_densification/point_cloud/sample_point_cloud.las'
mesh_file = '../data/mini_spiez/2_densification/3d_mesh/2020_09_17_spiez_simplified_3d_mesh.fbx'

def main():
    #points, f = load_point_cloud()
    #visualize_point_cloud(points, f)

    points, f = load_mesh()
    visualize_point_cloud(points, f)
    #visualize_example()

    print("works!")

def load_mesh():
    f = pv.read(mesh_file)
    print("point_format: ")
    pointformat = f.point_format
    for spec in f.point_format:
        print(spec.name)
    pass

def load_point_cloud(center = True):
    f = laspy.file.File(point_cloud_file, mode= 'r')
    print("point_format: ")
    pointformat = f.point_format
    for spec in f.point_format:
        print(spec.name)
    points = np.vstack((f.X, f.Y, f.Z)).transpose()
    if center:
        points_avg = np.mean(points, axis = 0)
        points = points - points_avg
        #print(points_avg)
    colors = f.red

    print(points)
    print(colors)
    f.close()
    return points, f

def visualize_mesh():

    pass

def visualize_point_cloud(points, file):
    bcpos = [(6.20, 3.00, 7.50),
         (0.16, 0.13, 2.65),
         (-0.28, 0.94, -0.21)]

    p = pv.Plotter()
    poly = pv.PolyData(points)

    #p.add_mesh(poly, show_edges=True, color='white')
    p.add_mesh(pv.PolyData(poly.points), color='red',
       point_size=3, render_points_as_spheres=True)
    p.camera_position = bcpos
    p.show()
    pass

def visualize_example():

    mesh = examples.load_hexbeam()
    print(type(mesh))
    bcpos = [(6.20, 3.00, 7.50),
         (0.16, 0.13, 2.65),
         (-0.28, 0.94, -0.21)]

    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, color='white')
    p.add_mesh(pv.PolyData(mesh.points), color='red',
       point_size=2, render_points_as_spheres=True)
    #p.camera_position = bcpos
    p.show(screenshot='beam_nodes.png')


if __name__ == "__main__":
    main()
