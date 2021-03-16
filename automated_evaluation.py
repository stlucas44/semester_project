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

#files
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_file = data_folder + "/bunny.ply"
vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rone_file =  data_folder + "/rhone_enu.off"
gorner_file = data_folder + "/gorner.off"

# settings:
speed = 0 # 0 for high sensor resolution,
plot_subplots = True
show_subplots = False

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



def main(params,  aic = False):

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
                                  look_down = look_down)

    view_point_angle = get_vp(rpy)
    # sample it for comparison and update
    measurement_pc = sample_points(true_mesh, n_points = n_pc_measurement)

    measurement_gmm = Gmm()
    measurement_gmm.pc_hgmm(measurement_pc, recompute = recompute_items, path = tmp_gmm_true_pc,
                            min_points = 500, cov_condition = cov_condition)
    save_to_file(measurement_gmm, get_file_path(params, "measurement", ".csv"))



    #measurement_gmm.pc_simple_gmm(measurement_pc, path = tmp_gmm_true_pc, recompute = True)
    if plot_subplots:
        mpl_subplots((measurement_pc, measurement_gmm), cov_scale = 2.0,
                     view_angle = view_point_angle,
                     path = get_figure_path(params, "measurement"),
                     title = ("measurement pc", "measurement gmm"),
                     show = show_subplots)
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
                 path = get_figure_path(params, "prior"),
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
    save_to_file(final_gmm, get_file_path(params, "final", ".csv"))


    if plot_result:
        mpl_visualize(final_gmm, title="final gmm", cov_scale = 2.0)

    if plot_subplots:
        mpl_subplots((true_mesh, final_gmm), cov_scale = 2.0,
                 view_angle = view_point_angle,
                 path = get_figure_path(params, "final"),
                 title = ("true_mesh", "final gmm"),
                 show = show_subplots)
        mpl_subplots((prior_mesh, final_gmm), cov_scale = 2.0,
                 view_angle = view_point_angle,
                 path = get_figure_path(params, "comp"),
                 title = ("prior mesh", "final gmm"),
                 show = show_subplots)
        plt.close('all')

    # Free memory:
    prior_mesh = None

    #### compute scores
    # score the corrupted gmm with sampled mesh
    print('Starting scoring'.center(80,'*'))
    true_pc = sample_points(true_mesh, n_points = n_pc_true) #n_pc_true)

    score_prior = evaluation.eval_quality_maha(prior_gmm, true_pc)
    score_merged = evaluation.eval_quality_maha(final_gmm, true_pc)

    # generate gmms
    true_gmm = Gmm()
    #true_gmm.mesh_gmm(true_mesh, n = len(measurement_gmm.means), recompute = recompute_items, path = tmp_gmm_true)
    #true_gmm.naive_mesh_gmm(true_mesh, mesh_std = 0.05)
    true_gmm.mesh_hgmm(true_mesh,  min_points = 8,
                      max_mixtures = 800,
                      verbose = False,
                      cov_condition = cov_condition)
    save_to_file(true_gmm, get_file_path(params, "true", ".csv"))
    score_true = evaluation.eval_quality_maha(true_gmm, true_pc)

    print("Maha Scores: true, prior, updated", score_true, score_prior, score_merged)
    plt.close('all')

    if aic:
        aic_true = evaluation.eval_quality_AIC(true_gmm, true_pc)
        aic_prior = evaluation.eval_quality_AIC(prior_gmm, true_pc)
        aic_merged = evaluation.eval_quality_AIC(final_gmm, true_pc)

        print("AIC Scores: true, prior, updated", aic_true, aic_prior, aic_merged)
        aic_values = [[aic_true, aic_prior, aic_merged]]
        p_values =  evaluation.compare_AIC(aic_values)
        print("AIC P values: ", p_values)
        return ((score_true, score_prior, score_merged), p_values)

    return score_true, score_prior, score_merged

if __name__ == "__main__":

    #settings:
    bunny_mesh_params = {"path" : bunny_file, "aag" : (1.0, 3.0), "pc_sensor_fov" : [100, 85],
                         "disruption_range" : (0.0, 0.5),
                         "disruption_patch_size" : 0.15,
                         "refit_voxel_size" : 0.01,
                         "cov_condition" : 0.02,
                         "cov_condition_resampling" : 0.02,
                         "corruption_percentage" : 0.2
                         }
    curve_mesh_params = {"path" : curve_file, "aag" : (2.0,4.0), "pc_sensor_fov" : [80, 85],
                         "disruption_range" : (1.0, 2.0),
                         "disruption_patch_size" : 1.0,
                         "refit_voxel_size": 0.1,
                         "cov_condition" : 0.1,
                         "cov_condition_resampling" : 0.15,
                         "corruption_percentage" : 0.2
                         }

    params_list = [bunny_mesh_params, curve_mesh_params]


    corruptions = [0.05, 0.1, 0.2, 0.4]
    iterations_per_scale = 5
    results = np.zeros((len(corruptions), iterations_per_scale, 3))
    for params in params_list:
        for (scale_number, corruption_scale) in zip(np.arange(0,len(corruptions)),corruptions):
            for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
                print(("Starting on current scale: " + str(corruption_scale) +
                       " current iteration: " + str(iteration)).center(80,'*'))
                params["corruption_percentage"] = corruption_scale
                result = main(params) # result is [1,3]
                #print("worked again")
                results[scale_number, iteration] = result
                gc.collect()
                #print("results: ", results)
        labels = ["True", "Prior", "Refined"]
        draw_advanced_box_plots(results, labels, corruptions,
                                title = "Evaluation wrt prior quality (n = " + str(iterations_per_scale),
                                path = get_figure_path(params, "box"),
                                show = False)

        print(("finished with" + params['path']).center(100, '*'))
        gc.collect()
