import numpy as np
import matplotlib

import open3d as o3

import pyvista as pv

from pyvista import examples

import laspy

verbosity = 0
point_cloud_file= '../data/mini_spiez_2/2_densification/point_cloud/2020_09_17_spiez_group1_densified_point_cloud.las'
mesh_file = '../data/mini_spiez_2/2_densification/3d_mesh/2020_09_17_spiez_simplified_3d_mesh.obj'
pc_bunny = 'bunny.pcd'

def main():
    poly= load_point_cloud_las(point_cloud_file, center = True)
    visualize_point_cloud(poly)

    mesh = load_mesh(mesh_file)
    visualize_mesh(mesh)

    #visualize_example()

    print("works!")

def load_mesh(mesh_file):
    f = pv.read(mesh_file)
    return f
    dprint("point_format: ")
    #pointformat = f.point_format
    for spec in f.point_format:
        dprint(spec.name)
    pass


def load_point_cloud_las(pc_file, v = 2, center = True):
    #load
    f = laspy.file.File(pc_file, mode= 'r')
    #print points
    dprint("point_format: ")
    pointformat = f.point_format
    for spec in f.point_format:
        dprint(spec.name)
    #reshape
    points = np.vstack((f.X, f.Y, f.Z)).transpose()
    dprint(np.size(points))
    #center
    if center:
        points_avg = np.mean(points, axis = 0)
        points = points - points_avg
        #print(points_avg)

    #advanced settings:
    I = f.Classification == 2
    dprint("Classification:" + str(f.Intensity))
    Points_bla = np.c_[f.X, f.Y, f.Z]
    Points_bla

    poly = pv.PolyData(points)
    f.close()

    return poly
def load_point_cloud_pcl(pc_file, v= 2, center = True):
    source = o3.io.read_point_cloud(pc_file)


    pass

def visualize_mesh(mesh):
    p = pv.Plotter()

    p.add_mesh(mesh, color='red',
        point_size=3, render_points_as_spheres=True)
    p.add_floor(face='-z', i_resolution=10, j_resolution=10,
        color=None, line_width=None, opacity=0.5,
        show_edges=False, lighting=False, edge_color=None,
        reset_camera=None, pad=0.0, offset=0.0, pickable=False, store_floor_kwargs=True)
    p.show()
    pass

def visualize_point_cloud(poly):
    scale = 0.5
    bcpos = [10.0, 10.0, 10.0]
    bcpos = np.multiply(scale,bcpos)

    p = pv.Plotter()

    #p.add_mesh(poly, show_edges=True, color='white')
    p.add_mesh(pv.PolyData(poly.points), color='red',
       point_size=3, render_points_as_spheres=True)
    p.camera_position = bcpos
    p.add_axes_at_origin(x_color="blue", y_color="blue", z_color="green",
        #xlabel='X', ylabel='Y', zlabel='Z',
        line_width=5, labels_off=True)
    p.add_floor(face='-z', i_resolution=10, j_resolution=10,
        color=None, line_width=None, opacity=0.5,
        show_edges=False, lighting=False, edge_color=None,
        reset_camera=None, pad=0.0, offset=0.0, pickable=False, store_floor_kwargs=True)
    p.show()
    #p.show_axes()
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

def dprint(content):
    if verbosity > 0:
        print(content)
    else:
        pass

if __name__ == "__main__":
    main()
