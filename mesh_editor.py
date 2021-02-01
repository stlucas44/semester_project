import copy
import open3d as o3d
from os.path import expanduser
from lib import visualization as vis
from lib.loader import *

import networkx as nx
import trimesh

'''
Use this editor to edit, generate, scale meshes
Cases:
1 Create cube
2 crush bunny
3 scale mesh and pc
4 (TO BE implented): create a measurement with a mesh (glacier or else)
'''

edit_case = 3
gen_cube = False
crush_bunny = False
scale_bunny = False
crush_random = True

model_scaling = 10


home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_mesh_file_large = data_folder + "/bunny/reconstruction/bun_zipper_res4_large.ply"
bunny_mesh_file_large_corr = data_folder + \
                        "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"

bunny_pc_file = data_folder + "/bunny/data/bun045.ply"
bunny_pc_file_large = data_folder + "/bunny/data/bun045_large.ply"




def main():
    if gen_cube:
        local_cube = cubeGenerator()
        vis.visualize_mesh(local_cube.cube, linewidth = 1.0, show = True)

    if crush_bunny:
        # load bunny and scaling up
        bunny_mesh = load_mesh(bunny_mesh_file, scale = model_scaling)
        new_bunny = corrupt_region_volumetric(bunny_mesh, region_center = [-0.25, 1.0, 0.2],
                                        offset = [0.0, 0.0, 0.3])
        vis.mpl_visualize(new_bunny)
        o3d.io.write_triangle_mesh(bunny_mesh_file_large_corr, new_bunny)

    if scale_bunny:
        bunny = load_mesh(bunny_mesh_file, scale = model_scaling)
        o3d.io.write_triangle_mesh(bunny_mesh_file_large, bunny)

        bunny_pc = load_measurement(bunny_pc_file, scale = model_scaling)
        o3d.io.write_point_cloud(bunny_pc_file_large, bunny_pc)

    if crush_random:
        bunny = load_mesh(bunny_mesh_file, scale = model_scaling)
        bunny = corrupt_region_connected(bunny, corruption_percentage = 0.5, offset_range = (1.0, 1.5))

        vis.mpl_visualize(bunny, alpha = 0.8)

    pass

class cubeGenerator():
    def __init__(self, w = 1.0, h = 1.0 , d = 1.0):
        self.w = w
        self.h = h
        self.d = d
        self.cube = self.generate_cube()

    def generate_cube(self):
        return o3d.geometry.TriangleMesh.create_box(width = self.w,
                                                    height = self.h,
                                                    depth = self.d)

    def save(self, name, path):
        o3d.io.write_triangle_mesh(name + ".ply", self.cube)


def corrupt_region_volumetric(mesh,
                 region_center = [0.0, 0.0, 0.0],
                 region = 0.3,
                 offset = [0.0, 0.0, 0.0]):
    points = np.asarray(mesh.vertices)
    local_points_index = [i for i, point in enumerate(points) \
                          if np.linalg.norm(point - region_center) < region]
    #print(local_points_index[:10])

    offset = offset * np.asarray([0.0, 0.0, 1.0])
    points[local_points_index, :] = points[local_points_index, :] + np.asarray(offset)

    mesh.vertices = o3d.utility.Vector3dVector(points)
    return mesh

def corrupt_region_connected(mesh, corruption_percentage = 0.1,
                             n_points = 1,
                             offset_range = (-0.5,0.5),
                             max_batch_area = 0.3):

    print(type(mesh))

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    #transform to trimesh
    local_mesh = trimesh.base.Trimesh(vertices = mesh.vertices,
                                      faces = mesh.triangles,
                                      face_normals = mesh.triangle_normals,
                                      vertex_normals = mesh.vertex_normals)

    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    #print(local_points_index[:10])
    for i in np.arange(0, n_points):
        init_index = [np.random.randint(0, len(mesh.vertices))]
        expected_size = np.random.uniform(low = 0, high = max_batch_area)
        print(type(init_index))

        #get all points that are connected to this
        area = 0.0
        while area < expected_size:
            #neighboring_mask = [init_index in line for line in faces]
            neighboring_mask = [any(elem in init_index for elem in line) for line in faces]
            neighboring_vertices = [i for i, x in enumerate(neighboring_mask) if x]
            print(np.asarray(neighboring_vertices).shape)
            neighboring_index = np.asarray(faces[neighboring_mask]).reshape(-1,1)
            neighboring_index = np.unique(neighboring_index)
            print(neighboring_index)
            area = np.asarray(local_mesh.area_faces)[neighboring_mask].sum()
            print("  pertubed_mesh_area: ", area)
            init_index = neighboring_index

        mask = np.zeros((len(points)), dtype = bool)
        mask[neighboring_index] = True
        mask[init_index] = True

        offset_length = np.random.uniform(low = offset_range[0], high = offset_range[1])
        print("  mesh pertubed by ", offset_length)
        offset = np.asarray(mesh.vertex_normals)[init_index] * offset_length

        points[mask, :] = points[mask, :] + np.asarray(offset)
        if len(neighboring_index)/len(mask) > corruption_percentage:
            break
        print("  corrupted vertex precentage: ", len(neighboring_index)/len(mask))

    print("finished corruption")
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def remove_region(mesh, region_center, region):
    print("implement REMOVE REGION!")
    pass

if __name__ == "__main__":
    main()
