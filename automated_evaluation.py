import copy
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
vicon_name = vicon_file.rfind("/")
curve_file = data_folder + "/curve.off"

def get_name(name):
    start = name.rfind("/") + 1
    end = name.rfind(".")
    return name[start:end]


speed = 1 # 0 for high sensor resolution,
plot = False
# sensor params:
if speed == 0:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]  # aligned with what?
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 0.3 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e6
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



def main(path, corruption_percentage, altitude_above_ground, pc_sensor_fov):
    #### true mesh
    print('Prepare true mesh and pc'.center(80,'*'))
    # load true mesh
    true_mesh = automated_view_point_mesh(path, altitude_above_ground = (1.0, 3.0),
                                  sensor_fov = pc_sensor_fov,
                                  #sensor_max_range = 100.0,
                                  angular_resolution = 1.0)

    # sample it for comparison and update
    true_pc = sample_points(true_mesh, n_points = n_pc_true) #n_pc_true)
    measurement_pc = sample_points(true_mesh, n_points = n_pc_measurement)

    measurement_gmm = Gmm()
    measurement_gmm.pc_hgmm(measurement_pc, recompute = recompute_items, path = tmp_gmm_true_pc)
    #measurement_gmm.pc_simple_gmm(measurement_pc, path = tmp_gmm_true_pc, recompute = True)

    # generate gmms
    true_gmm = Gmm()
    true_gmm.mesh_gmm(true_mesh, n = len(measurement_gmm.means), recompute = recompute_items, path = tmp_gmm_true)
    #true_gmm.naive_mesh_gmm(true_mesh, mesh_std = 0.05)


    #### corrupted mesh
    print('Prior mesh'.center(80,'*'))

    # load corrupted mesh
    prior_mesh = corrupt_region_connected(true_mesh, corruption_percentage = 0.2,
                                 n_max = 10,
                                 offset_range = (-0.5,0.5),
                                 max_batch_area = 0.15)

    # generate gmm from prior
    prior_gmm = Gmm()
    #prior_gmm.mesh_gmm(prior_mesh, n = len(prior_mesh.triangles), recompute = recompute_items, path = tmp_gmm_prior)
    prior_gmm.mesh_gmm(prior_mesh, n = len(measurement_gmm.means), recompute = recompute_items, path = tmp_gmm_prior)
    #prior_gmm.naive_mesh_gmm(prior_mesh, mesh_std = 0.05)


    #### merge mesh with corrupted point cloud
    # apply merge with
    print('Compute merge'.center(80,'*'))

    merged_gmm_lists, final_gmm, final_gmm_pair = merge.gmm_merge(
            prior_gmm,
            measurement_gmm,
            p_crit = 0.05,
            sample_size = 5,
            n_resample = n_resampling)
    if plot:
        mpl_visualize(final_gmm, title="final gmm", cov_scale = 2.0)
        mpl_visualize(*final_gmm_pair, colors = ["g", "r"],
                      cov_scale = 2.0, show_mean = False,
                      view_angle = view_point_angle, show_z = False,
                      title = "final pair")

    #### compute scores
    # score the corrupted gmm with sampled mesh
    print('Starting scoring'.center(80,'*'))

    score_true = evaluation.eval_quality_maha(true_gmm, true_pc)
    score_prior = evaluation.eval_quality_maha(prior_gmm, true_pc)
    score_merged = evaluation.eval_quality_maha(final_gmm, true_pc)

    print("Scores: true, prior, updated", score_true, score_prior, score_merged)

    return score_true, score_prior, score_merged

    '''
    score_true = evaluation.eval_quality(true_gmm, true_pc)
    score_prior = evaluation.eval_quality(prior_gmm, true_pc)
    score_merged = evaluation.eval_quality(final_gmm, true_pc)

    print("Scores: true, prior, updated", score_true, score_prior, score_merged)
    '''

if __name__ == "__main__":

    corruption_part = [0.05, 0.1, 0.2, 0.5]
    iterations_per_scale = 5
    results = np.zeros((iterations_per_scale, 3))

    # variables: bunny: 0-5 - 1.0, curve = 5-10
    files = [bunny_mesh_file, curve_file, vicon_file]
    altitude_above_ground = (5.0,10.0)
    pc_sensor_fov = [100, 85]

    print(get_name(curve_file))

    corruption_scale = 0.2
    #for corruption_scale in corruptions:
    for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
        result = main(curve_file, corruption_scale, altitude_above_ground, pc_sensor_fov)
        print("worked again")
        results[iteration] = result
        #print("results: ", results)

    labels = ["True", "Prior", "Refined"]
    draw_box_plots(results, labels, title = "Dataset: " + get_name(curve_file))
