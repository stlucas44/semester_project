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
crush_random = False
load_random = False
reduce_mesh = True

model_scaling = 10


home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_mesh_file_large = data_folder + "/bunny/reconstruction/bun_zipper_res4_large.ply"
bunny_mesh_file_large_corr = data_folder + \
                        "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"

bunny_pc_file = data_folder + "/bunny/data/bun045.ply"
bunny_pc_file_large = data_folder + "/bunny/data/bun045_large.ply"

spiez_file = data_folder + "/spiez.obj"
rhone_file = data_folder + "/rhone_enu.off"
huenliwald_file = data_folder + "/huenliwald.off"
bunny_file = data_folder +  "/bunny.ply"


spiez_region = ((-20.0, -30.0), (5.0, 5.0))
rohne_region = ((200.0, -250.0), (350.0,-50.0))
huenliwald_region = ((0.0, 0.0), (100.0, 100.0))


path = bunny_file
region = huenliwald_file

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
        bunny = corrupt_region_connected(bunny, corruption_percentage = 0.2, offset_range = (0.1, 0.3))
        vis.mpl_visualize(bunny, alpha = 0.8)

    if load_random:
        bunny = automated_view_point_mesh(bunny_mesh_file_large, sensor_max_range = 3.0)
        vis.mpl_visualize(bunny)

    if subsample_mesh:
        vp = (90.0, 0.0)
        mesh = o3d.io.read_triangle_mesh(path)
        vis.mpl_visualize(mesh, view_angle = vp)
        mesh = select_local_region(mesh, region, axes = 'xy')
        #mesh = sub_sample
        vis.mpl_visualize(mesh, alpha = 1.0,  view_angle = vp)
        save_submesh(mesh, path)
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
                             n_max = 10,
                             offset_range = (-0.5,0.5),
                             max_batch_area = 0.3,
                             check_intersection = False,
                             smoothness = 0.0,
                             verbose = False):


    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    new_mesh = copy.deepcopy(mesh)


    #transform to trimesh
    local_mesh = trimesh.base.Trimesh(vertices = mesh.vertices,
                                      faces = mesh.triangles,
                                      face_normals = mesh.triangle_normals,
                                      vertex_normals = mesh.vertex_normals)

    points = np.asarray(new_mesh.vertices)
    faces = np.asarray(new_mesh.triangles)
    overall_neighboring_index = np.empty((0,))
    #print(local_points_index[:10])
    iterations = 0
    for i in np.arange(0, n_max):
        init_index = [np.random.randint(0, len(mesh.vertices))]
        expected_size = np.random.uniform(low = 0, high = max_batch_area)
        neighboring_index = None
        #get all points that are connected to this
        area = 0.0
        if verbose: print("init_index before start: ", init_index)
        for j in np.arange(0, n_max):
            neighboring_mask = [any(elem in init_index for elem in line) for line in faces]
            neighboring_vertices = [k for k, x in enumerate(neighboring_mask) if x]

            neighboring_index = np.asarray(faces[neighboring_mask]).reshape(-1,1)
            neighboring_index = np.unique(neighboring_index)
            #print(neighboring_index)
            area = np.asarray(local_mesh.area_faces)[neighboring_mask].sum()
            if verbose: print("  pertubed_mesh_area: ", area)
            if area > expected_size:
                break

            init_index = np.concatenate((init_index, neighboring_index))### TODO: FIX THIS SHIT!

        mask = np.zeros((len(points)), dtype = bool)
        mask[neighboring_index] = True
        mask[init_index] = True

        offset_length = np.random.uniform(low = offset_range[0], high = offset_range[1])
        offset = np.asarray(mesh.vertex_normals)[init_index[0]] * offset_length
        #print("  mesh pertubed by ", offset_length)

        #print(np.asarray(init_index).shape)
        #print(points.shape)
        #dist = [points[init_index[0]] - points[init_index[:]]]
        #print(dist[1:20])
        #offset_weight = np.linalg.norm(dist, axis = 1)

        points[mask, :] = points[mask, :] + np.asarray(offset)

        overall_neighboring_index = np.concatenate((overall_neighboring_index, neighboring_index))
        if verbose: print("  corrupted vertex precentage: ", len(overall_neighboring_index)/len(mask))

        iterations = i
        if len(overall_neighboring_index)/len(mask) > corruption_percentage:
            print("exited by corruption percentage")
            break


    if verbose: print("finished corruption")

    new_mesh.vertices = o3d.utility.Vector3dVector(points)
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()

    self_intersection = new_mesh.is_self_intersecting()

    # remove triangles?
    if self_intersection and check_intersection:
        print("selfintersection!!")
        return None
    elif self_intersection:
        print("selfintersection!!")

    return new_mesh, iterations

def remove_region(mesh, region_center, region):
    print("implement REMOVE REGION!")
    pass


def subsample_mesh(mesh, n_triangles = 20000):
    mesh = o3d.io.read_triangle_mesh(path)


    if len(mesh.triangles) > 20000:
        print(" mesh to large, subsampling!")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangles)
    mesh.compute_vertex_normals()

    return mesh

def select_local_region(mesh, region_boundaries = ((0.0, 0.0),(1.0, 1.0)), axes = 'xy'):
    region_boundaries = np.asarray(region_boundaries)

    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    triangle_points = points[triangles]

    print(triangle_points.shape)

    if axes == 'xy':
        triangle_points = triangle_points[:,:,[0,1]]
    elif axes == 'yz':
        triangle_points = triangle_points[:,:,[1,2]]
    elif axes == 'xz':
        triangle_points = triangle_points[:,:,[0,2]]

    in_region_mask = np.zeros((len(triangles),), dtype = bool)
    print("Region boundaries: ", region_boundaries)

    for (i, point_triplet) in enumerate(triangle_points):
        in_region = np.asarray([region_boundaries[0] < point_triplet[0],
                    point_triplet[0] < region_boundaries[1],
                    region_boundaries[0]< point_triplet[1],
                    point_triplet[1] < region_boundaries[1],
                    region_boundaries[0]< point_triplet[2],
                    point_triplet[2] < region_boundaries[1]])
        in_region = in_region.reshape((-1,))
        in_region_mask[i] = in_region.all()
        #print(in_region.all())

    not_in_region_mask = [not x for x in in_region_mask]

    mesh.remove_triangles_by_mask(not_in_region_mask)
    mesh.remove_unreferenced_vertices()

    print("Now ", in_region_mask.sum(), " triangles")
    return mesh

def save_submesh(local_mesh, path, key_word = "_reduced"):
    dot_loc = path.find(".")
    new_path = path[:dot_loc] + key_word + path[dot_loc:]

    o3d.io.write_triangle_mesh(new_path, local_mesh)

if __name__ == "__main__":
    main()
