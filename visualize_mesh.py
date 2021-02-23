from os.path import expanduser


from lib import visualization
from lib import loader


home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045_large.ply"

#starting in the data folder
'''
path_collection = ["/mini_spiez_2/2_densification/3d_mesh/2020_09_17_spiez_simplified_3d_mesh.obj",
                   "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply",
                   "/bunny/reconstruction/bun_zipper_res4_large.ply",
                   "/rhone_enu.off",
                   "/vicon.stl"]
path_collection = [data_folder + path for path in path_collection]

'''
bunny_file = data_folder + "/bunny.ply"
vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rhone_file =  data_folder + "/rhone_enu_reduced.off"
huenli_file = data_folder + "/gorner.off"
spiez_file = data_folder + "/spiez_reduced.obj"

path_collection = [bunny_file, vicon_file, curve_file, rhone_file, spiez_file]
print(path_collection)

for path in path_collection:
    mesh = loader.load_mesh(path)
    visualization.o3d_visualize(mesh)
    #visualization.mpl_visualize(mesh, alpha = 1.0)
print("worked")
