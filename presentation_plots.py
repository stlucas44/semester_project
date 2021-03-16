import copy
import gc
import open3d as o3d
from os.path import expanduser

from lib import evaluation
from lib import registration
from lib import merge

from lib.gmm_generation import Gmm, save_to_file
from lib.loader import *
from lib.visualization import *

from mesh_editor import corrupt_region_connected
# settings:
speed = 0 # 0 for high sensor resolution,
plot_subplots = True
show_subplots = True

# Single plots:
plot_sensor = False
plot_match = False
plot_result = False


# sensor params:
if speed == 0:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]  # aligned with what?
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 0.3 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e5
    n_pc_measurement = 1e5
    n_resampling = int(1e5)

# goal: plot nice gmm for bunny, curve, spiez and rohne
# define sensor positions
vp_bunny = (1.0, 2.0, 10.0) # (1.0, 2.0, 1.0) best so far
vp_vis_bunny = (90.0, -90.0)
vp_curve = (35.0, -3.0, 0.0)
vp_vis_curve = (30.0, -10.0)
vp_spiez = (5.0, 5.0, -18.0)
vp_vis_spiez = (30.0, 20.0)
vp_rhone = (200.0, -150.0, 20.0)
vp_vis_rohne = (70.0, 0.0)


# define paths
#paths:
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_file = data_folder + "/bunny.ply"
vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rhone_file =  data_folder + "/rhone_enu_reduced.off"
huenli_file = data_folder + "/gorner.off"
spiez_file = data_folder + "/spiez_reduced.obj"


#settings:
bunny_mesh_params = {"path" : bunny_file, "aag" : (1.0, 3.0), "pc_sensor_fov" : [180, 180],
                     "disruption_range" : (0.0, 0.3),
                     "disruption_patch_size" : 0.15,
                     "refit_voxel_size" : 0.01,
                     "cov_condition" : 0.02,
                     "cov_condition_resampling" : 0.04,
                     "corruption_percentage" : 0.2,
                     "look_down" : False
                     }

curve_mesh_params = {"path" : curve_file, "aag" : (2.0,4.0), "pc_sensor_fov" : [80, 85],
                     "disruption_range" : (0.5, 2.0),
                     "disruption_patch_size" : 1.0,
                     "refit_voxel_size": 0.1,
                     "cov_condition" : 0.1,
                     "cov_condition_resampling" : 0.1,
                     "corruption_percentage" : 0.2,
                     "look_down" : False
                     }

rhone_params = {"path" : rhone_file, "aag" : (50.0, 100.0), "pc_sensor_fov" : [80, 60],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.05,
                "cov_condition_resampling" : 0.1,
                "corruption_percentage" : 0.2,
                "look_down" : False
                }

spiez_params = {"path" : spiez_file, "aag" : (0.5, 1.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.8,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.2,
                "cov_condition_resampling" : 0.2,
                "corruption_percentage" : 0.2,
                "look_down" : False
                }

aic = False



# vp
#view_point_angle =  (45.0, 45.0)
view_point_angle =  (0.0, 0.0)

#precomputed gmms:
recompute_items = True
tmp_gmm_true = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_true_pc = data_folder + "/tmp/tmp_measurement_gmm_pc"
tmp_gmm_prior = data_folder + "/tmp/tmp_mesh_gmm"


def main(params,  aic = False, vp = None, vp_vis = None):

    path = params['path']
    altitude_above_ground = params['aag']
    pc_sensor_fov = params['pc_sensor_fov']
    refit_voxel_size = params['refit_voxel_size']
    disuption_range = params["disruption_range"]
    disruption_patch_size = params["disruption_patch_size"]
    cov_condition = params["cov_condition"]
    cov_condition_resampling = params["cov_condition_resampling"]
    corruption_percentage = params["corruption_percentage"]
    look_down = params["look_down"]



    #### true mesh
    print('Prepare true mesh and pc'.center(80,'*'))

    # load true mesh
    true_mesh, rpy = automated_view_point_mesh(path, altitude_above_ground = altitude_above_ground,
                                  sensor_fov = pc_sensor_fov,
                                  #sensor_max_range = 100.0,
                                  angular_resolution = 1.0,
                                  plot = plot_sensor,
                                  only_important = True,
                                  look_down = look_down,
                                  vp = vp)
    if vp_vis is not None:
        view_point_angle = vp_vis

    elif vp is not None:
        view_point_angle = get_vp(rpy, offset = vp)
    else:
        view_point_angle = get_vp(rpy)

    # sample it for comparison and update
    measurement_pc = sample_points(true_mesh, n_points = n_pc_measurement)
    #o3d_visualize(measurement_pc, mesh = False)


    measurement_gmm = Gmm()
    measurement_gmm.pc_hgmm(measurement_pc, recompute = recompute_items, path = tmp_gmm_true_pc,
                            min_points = 500, cov_condition = cov_condition)
    #save_to_file(measurement_gmm, get_file_path(params, "measurement", ".csv"))



    #measurement_gmm.pc_simple_gmm(measurement_pc, path = tmp_gmm_true_pc, recompute = True)
    if plot_subplots:
        mpl_subplots((measurement_pc, measurement_gmm), cov_scale = 2.0,
                     view_angle = view_point_angle,
                     path = get_figure_path(params, "measurement", folder = "imgs/presentation_plots/"),
                     title = ("measurement pc", "measurement gmm"),
                     show = show_subplots,
                     show_z = False)
        plt.close('all')

    # Free memory
    measurement_pc = None

    #### corrupted mesh
    print('Prior mesh'.center(80,'*'))

    # load corrupted mesh
    prior_mesh, num_disruptions = corrupt_region_connected(true_mesh, corruption_percentage = corruption_percentage,
                                 n_max = 10,
                                 offset_range = disuption_range,
                                 max_batch_area = disruption_patch_size)

    if plot_subplots:
        #visualization.o3d_visualize(true_mesh)
        #visualization.o3d_visualize(prior_mesh)

        mpl_subplots((true_mesh, prior_mesh), cov_scale = 2.0,
                     view_angle = view_point_angle,
                     path = get_figure_path(params, "mesh_compare", folder = "imgs/presentation_plots/"),
                     title = ("true_mesh", "prior_mesh"),
                     show = show_subplots,
                     show_z = False)
        plt.close('all')

    # generate gmm from prior
    prior_gmm = Gmm()
    #prior_gmm.mesh_gmm(prior_mesh, n = len(prior_mesh.triangles), recompute = recompute_items, path = tmp_gmm_prior)
    #prior_gmm.mesh_gmm(prior_mesh, n = len(measurement_gmm.means + num_disruptions), recompute = recompute_items, path = tmp_gmm_prior)
    prior_gmm.mesh_hgmm(prior_mesh,  min_points = 8,
                      max_mixtures = 800,
                      verbose = False,
                      cov_condition = cov_condition)

    if plot_subplots:
        mpl_subplots((prior_mesh, prior_gmm), cov_scale = 2.0,
                 view_angle = view_point_angle,
                 path = get_figure_path(params, "prior", folder = "imgs/presentation_plots/"),
                 title = ("prior mesh", "prior gmm"), show = show_subplots)
        plt.close('all')


    #### merge mesh with corrupted point cloud
    # apply merge with
    print('Compute merge'.center(80,'*'))

    final_gmm = merge.gmm_merge(
            prior_gmm,
            measurement_gmm,
            p_crit = 0.05,
            sample_size = 5,
            n_resample = n_resampling,
            plot = plot_match,
            refit_voxel_size = refit_voxel_size,
            cov_condition = cov_condition_resampling,
            return_only_final = True)

    merged_gmm_lists = None
    final_gmm_pair = None
    # save to file!
    #save_to_file(final_gmm, get_file_path(params, "final", ".csv"))


    if plot_result:
        mpl_visualize(final_gmm, title="final gmm", cov_scale = 2.0)

    if plot_subplots:
        mpl_subplots((prior_gmm, final_gmm), cov_scale = 2.0,
                 view_angle = view_point_angle,
                 path = get_figure_path(params, "final", folder = "imgs/presentation_plots/"),
                 title = ("prior gmm", "final gmm"),
                 show = show_subplots,
                 show_z = False)

        mpl_subplots((prior_mesh, final_gmm), cov_scale = 2.0,
                 view_angle = view_point_angle,
                 path = get_figure_path(params, "comp", folder = "imgs/presentation_plots/"),
                 title = ("prior mesh", "final gmm"),
                 show = show_subplots)
        plt.close('all')

    # Free memory:
    prior_mesh = None

if __name__ == "__main__":
    #settings:
    vps = [vp_bunny, vp_curve, vp_rhone, vp_spiez]
    param_list = [bunny_mesh_params, curve_mesh_params, spiez_params, rhone_params]

    vps = [vp_curve]
    param_list = [curve_mesh_params]
    vp_viss = [vp_vis_curve]

    vps = [vp_rhone]
    param_list = [rhone_params]
    vp_viss = [vp_vis_rohne]

    vps = [vp_spiez]
    param_list = [spiez_params]
    vp_viss = [vp_vis_spiez]

    #vps = [vp_bunny]
    #param_list = [bunny_mesh_params]
    #vp_viss = [vp_vis_bunny]

    #vps = [vp_spiez, vp_curve]
    #param_list = [spiez_params, curve_mesh_params]
    #vp_vis = [vp_vis_spiez, vp_vis_curve]

    for (param, vp, vp_vis) in zip(param_list, vps, vp_viss):
        print("inserting: ", param, vp)
        main(param, vp = vp, vp_vis = vp_vis)
        #main(param)
