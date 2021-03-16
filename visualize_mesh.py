from os.path import expanduser


from lib import visualization
from lib import loader

from lib import gmm_generation

home = expanduser("~")
data_folder = home + "/semester_project/src/data"
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

vis_mesh = True
vis_gmm = False

if vis_mesh:
    for path in path_collection:
        mesh = loader.load_mesh(path)
        #visualization.o3d_visualize(mesh)
        visualization.mpl_visualize(mesh, alpha = 1.0, view_angle = (30,20))
    print("worked")
if vis_gmm:
    simple_gmm = gmm_generation.Gmm()
    h_gmm = gmm_generation.Gmm()
    print("Loading: ", )
    mesh = loader.load_mesh(bunny_file)

    mesh = loader.view_point_crop_by_cast(mesh, [0.0, 1.0, 2.0],
                        [0.0, 89.0, 89.0],
                        sensor_max_range = 100.0,
                        sensor_fov = [180.0, 180.0],
                        angular_resolution = 1.0,
                        get_pc = False,
                        plot = False,
                        only_important = False)

    pc = loader.sample_points(mesh)
    simple_gmm.pc_simple_gmm(pc, n = 20)
    h_gmm.pc_hgmm(pc, cov_condition = 0.05)

    visualization.mpl_subplots((simple_gmm, h_gmm), cov_scale = 2.0,
             view_angle =  (-90,90),
             show_z = False,
             path = home + "/semester_project/src/imgs/thesis_plots/gmm_hgmm_comparison.png",
             title = ("fixed size GMM", "hierarhcial GMM"), show = True)

    visualization.mpl_visualize(simple_gmm, cov_scale = 2.0, view_angle = (-90,90))
    visualization.mpl_visualize(h_gmm, cov_scale = 2.0, view_angle = (-90,90))
