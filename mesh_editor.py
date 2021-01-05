import copy
import open3d as o3d
from os.path import expanduser
from lib import visualization as vis
from lib.loader import *

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
crush_bunny = True
scale_bunny = True

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
        new_bunny = corrupt_region(bunny_mesh, region_center = [-0.25, 1.0, 0.2],
                                        offset = [0.0, 0.0, 0.3])
        vis.mpl_visualize(new_bunny)
        o3d.io.write_triangle_mesh(bunny_mesh_file_large_corr, new_bunny)

    if scale_bunny:
        bunny = load_mesh(bunny_mesh_file, scale = model_scaling)
        o3d.io.write_triangle_mesh(bunny_mesh_file_large, bunny)

        bunny_pc = load_measurement(bunny_pc_file, scale = model_scaling)
        o3d.io.write_point_cloud(bunny_pc_file_large, bunny_pc)

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


def corrupt_region(mesh,
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

def remove_region(mesh, region_center, region):
    print("implement REMOVE REGION!")
    pass

if __name__ == "__main__":
    main()
