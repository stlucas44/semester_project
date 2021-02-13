import copy
import gc
import open3d as o3d
from os.path import expanduser

from lib import evaluation
from lib import registration
from lib import merge

from lib.gmm_generation import Gmm
from lib.loader import *
from lib.visualization import *

from mesh_editor import corrupt_region_connected

#files
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large.ply"
bunny_mesh_file_corrupted = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"



vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rone_file =  data_folder + "/rhone_enu.off"
gorner_file = data_folder + "/gorner.off"


speed = 0 # 0 for high sensor resolution,
plot_sensor = True
plot_match = False
plot_result = False
plot_subplots = True
show_subplots = False

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

if speed == 1:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 3 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e4
    n_pc_measurement = 1e3
    n_resampling = 1e4


if speed == 2:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 5 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e4
    n_pc_measurement = 1e3
    n_resampling = 1e3



#precomputed gmms:
recompute_items = True
tmp_gmm_true = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_true_pc = data_folder + "/tmp/tmp_measurement_gmm_pc"
tmp_gmm_prior = data_folder + "/tmp/tmp_mesh_gmm"

#view point for visualization
view_point_angle =  (80.0, -60.0)



def main(params, corruption_percentage):

    path = params['path']
    altitude_above_ground = params['aag']
    pc_sensor_fov = params['pc_sensor_fov']
    refit_voxel_size = params['refit_voxel_size']
    disuption_range = params["disruption_range"]
    cov_condition = params["cov_condition"]
    #### true mesh
    print('Prepare true mesh and pc'.center(80,'*'))
    # load true mesh
    true_mesh = automated_view_point_mesh(path, altitude_above_ground = altitude_above_ground,
                                  sensor_fov = pc_sensor_fov,
                                  #sensor_max_range = 100.0,
                                  angular_resolution = 1.0,
                                  plot = plot_sensor)

    # sample it for comparison and update
    true_pc = sample_points(true_mesh, n_points = n_pc_true) #n_pc_true)
    measurement_pc = sample_points(true_mesh, n_points = n_pc_measurement)

    measurement_gmm = Gmm()
    measurement_gmm.pc_hgmm(measurement_pc, recompute = recompute_items, path = tmp_gmm_true_pc,
                            min_points = 500, cov_condition = cov_condition)
    #measurement_gmm.pc_simple_gmm(measurement_pc, path = tmp_gmm_true_pc, recompute = True)
    if plot_subplots:
        mpl_subplots((measurement_pc, measurement_gmm), cov_scale = 2.0,
                     path = get_figure_path(params, "measurement"),
                     title = ("measurement pc, measurement gmm"),
                     show = show_subplots)



    # generate gmms
    true_gmm = Gmm()
    #true_gmm.mesh_gmm(true_mesh, n = len(measurement_gmm.means), recompute = recompute_items, path = tmp_gmm_true)
    true_gmm.naive_mesh_gmm(true_mesh, mesh_std = 0.05)


    #### corrupted mesh
    print('Prior mesh'.center(80,'*'))

    # load corrupted mesh
    prior_mesh = corrupt_region_connected(true_mesh, corruption_percentage = corruption_percentage,
                                 n_max = 10,
                                 offset_range = disuption_range,
                                 max_batch_area = 0.15)

    # generate gmm from prior
    prior_gmm = Gmm()
    #prior_gmm.mesh_gmm(prior_mesh, n = len(prior_mesh.triangles), recompute = recompute_items, path = tmp_gmm_prior)
    prior_gmm.mesh_gmm(prior_mesh, n = len(measurement_gmm.means), recompute = recompute_items, path = tmp_gmm_prior)
    #prior_gmm.naive_mesh_gmm(prior_mesh, mesh_std = 0.05)
    if plot_subplots:
        mpl_subplots((prior_mesh, prior_gmm), cov_scale = 2.0,
                 path = get_figure_path(params, "prior"),
                 title = "prior mesh, prior gmm", show = show_subplots)

    #### merge mesh with corrupted point cloud
    # apply merge with
    print('Compute merge'.center(80,'*'))

    merged_gmm_lists, final_gmm, final_gmm_pair = merge.gmm_merge(
            prior_gmm,
            measurement_gmm,
            p_crit = 0.05,
            sample_size = 5,
            n_resample = n_resampling,
            plot = plot_match,
            refit_voxel_size = refit_voxel_size,
            cov_condition = cov_condition)
    if plot_result:
        mpl_visualize(final_gmm, title="final gmm", cov_scale = 2.0)
        mpl_visualize(*final_gmm_pair, colors = ["g", "r"],
                      cov_scale = 2.0, show_mean = False,
                      view_angle = view_point_angle, show_z = False,
                      title = "final pair")
    if plot_sensor:
        mpl_visualize(true_mesh, title = "true mesh")
        mpl_visualize(prior_mesh, title = "prior mesh")

    if plot_subplots:
        mpl_subplots((true_mesh, final_gmm), cov_scale = 2.0,
                 path = get_figure_path(params, "final"),
                 title = ("true_mesh, final gmm"),
                 show = show_subplots)

    #### compute scores
    # score the corrupted gmm with sampled mesh
    print('Starting scoring'.center(80,'*'))

    score_true = evaluation.eval_quality_maha(true_gmm, true_pc)
    score_prior = evaluation.eval_quality_maha(prior_gmm, true_pc)
    score_merged = evaluation.eval_quality_maha(final_gmm, true_pc)

    print("Maha Scores: true, prior, updated", score_true, score_prior, score_merged)

    #aic_true = evaluation.eval_quality_AIC(true_gmm, true_pc)
    #aic_prior = evaluation.eval_quality_AIC(prior_gmm, true_pc)
    #aic_merged = evaluation.eval_quality_AIC(final_gmm, true_pc)

    #print("AIC Scores: true, prior, updated", aic_true, aic_prior, aic_merged)
    plt.close('all')
    return score_true, score_prior, score_merged

if __name__ == "__main__":

    #settings:
    bunny_mesh_params = {"path" : bunny_mesh_file, "aag" : (1.0, 3.0), "pc_sensor_fov" : [100, 85],
                         "disruption_range" : (0.0, 0.5),
                         "refit_voxel_size" : 0.01,
                         "cov_condition" : 0.02}
    curve_mesh_params = {"path" : curve_file, "aag" : (3.0,6.0), "pc_sensor_fov" : [100, 85],
                         "disruption_range" : (0.5, 2.0),
                         "refit_voxel_size": 0.01,
                         "cov_condition" : 0.05}

    vicon_params = {"path" : vicon_file, "aag" : (3.0,6.0), "pc_sensor_fov" : [100, 85],
                         "disruption_range" : (0.5, 2.0),
                         "refit_voxel_size": 0.5,
                         "cov_condition" : 0.05}

    #params = curve_mesh_params
    params = curve_mesh_params


    corruptions = [0.05, 0.1, 0.2, 0.4]
    iterations_per_scale = 5
    results = np.zeros((len(corruptions), iterations_per_scale, 3))

    # variables: bunny: 0-5 - 1.0, curve = 5-10
    files = [bunny_mesh_file, curve_file, vicon_file]
    altitude_above_ground = (3.0,6.0)
    pc_sensor_fov = [100, 85]

    corruption_scale = 0.2
    for (scale_number, corruption_scale) in zip(np.arange(0,len(corruptions)),corruptions):
        for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
            print("** Starting on current scale:", corruption_scale, " current iteration:", iteration, " **")
            result = main(params, corruption_scale)
            #print("worked again")
            results[scale_number, iteration] = result
            gc.collect()
            #print("results: ", results)
    labels = ["True", "Prior", "Refined"]
    #draw_box_plots(results[0], labels, title = "Dataset: " + get_name(params['path']))
    draw_advanced_box_plots(results, labels, corruptions, path = get_figure_path(params, "box"),
                            show = False)
