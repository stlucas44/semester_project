import copy
import open3d as o3d
from os.path import expanduser

from lib import evaluation
from lib import registration
from lib import merge

from lib.gmm_generation import Gmm
from lib.loader import *
from lib.visualization import *

#files
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large.ply"
bunny_mesh_file_corrupted = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"

speed = 1 # 0 for high sensor resolution,

# sensor params:
if speed == 0:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 0.3 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e8
    n_pc_measurement = 1e6

if speed == 1:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 3 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e6
    n_pc_measurement = 1e4

if speed == 2:
    pc_sensor_position_enu = [0.0, 1.0, 2.0]
    pc_sensor_rpy = [0.0, 90.0, 90.0]
    pc_sensor_fov = [100, 85]
    pc_range = 6.0
    pc_angular_resolution = 5 # pico (354x287) would be 3.5 (res/fov)
    n_pc_true = 1e4
    n_pc_measurement = 1e3


#precomputed gmms:
recompute_items = False
tmp_gmm_true = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_true_pc = data_folder + "/tmp/tmp_measurement_gmm_pc"
tmp_gmm_prior = data_folder + "/tmp/tmp_mesh_gmm"

#view point for visualization
view_point_angle =  (80.0, -60.0)



def main():
    #### true mesh
    print('Prepare true mesh and pc'.center(80,'*'))
    # load true mesh
    #true_mesh = load_mesh(bunny_mesh_file)
    true_mesh = load_unit_mesh(type = '2trisA')

    # reduce to view point
    true_mesh_vp, true_mesh_occ = merge.view_point_crop(true_mesh,
           pc_sensor_position_enu,
           pc_sensor_rpy,
           sensor_max_range = pc_range,
           sensor_fov = pc_sensor_fov,
           angular_resolution = pc_angular_resolution)

    # sample it for comparison and update
    true_pc = sample_points(true_mesh_vp, n_points = n_pc_true) #n_pc_true)
    measurement_pc = sample_points(true_mesh_vp, n_points = n_pc_measurement)

    # generate gmms
    true_gmm = Gmm()
    true_gmm.mesh_gmm(true_mesh_vp, n = len(true_mesh.triangles), recompute = recompute_items, path = tmp_gmm_true)
    #true_gmm.naive_mesh_gmm(true_mesh_vp, mesh_std = 0.05)


    measurement_gmm = Gmm()
    measurement_gmm.pc_hgmm(measurement_pc, recompute = True, path = tmp_gmm_true_pc)
    #measurement_gmm.pc_simple_gmm(measurement_pc, path = tmp_gmm_true_pc, recompute = True)


    #### corrupted mesh
    print('True mesh and PC'.center(80,'*'))

    # load corrupted mesh
    #prior_mesh = load_mesh(bunny_mesh_file)
    #prior_mesh = load_mesh(bunny_mesh_file_corrupted)
    prior_mesh = load_unit_mesh(type = '2trisB')

    # view point crop
    prior_mesh_vp, prior_mesh_occ = merge.view_point_crop(prior_mesh,
           pc_sensor_position_enu,
           pc_sensor_rpy,
           sensor_max_range = pc_range,
           sensor_fov = pc_sensor_fov,
           angular_resolution = pc_angular_resolution)

    # generate gmm from prior
    prior_gmm = Gmm()
    prior_gmm.mesh_gmm(prior_mesh_vp, n = len(prior_mesh_vp.triangles), recompute = True, path = tmp_gmm_prior)
    #prior_gmm.naive_mesh_gmm(prior_mesh_vp, mesh_std = 0.05)


    #first plots:
    mpl_visualize(measurement_gmm, prior_gmm, colors = ["g", "r"], title="final gmm")
    mpl_visualize(measurement_pc, prior_mesh, colors = ["g", "r"], title="final gmm")


    #### merge mesh with corrupted point cloud
    # apply merge with
    print('Compute merge'.center(80,'*'))

    merged_gmm_lists, final_gmm, final_gmm_pair = merge.gmm_merge(
            prior_gmm,
            measurement_gmm,
            p_crit = 0.05,
            sample_size = 5)

    mpl_visualize(final_gmm, title="final gmm")
    mpl_visualize(*final_gmm_pair, colors = ["g", "r"],
                  cov_scale = 2.0, show_mean = False,
                  view_angle = view_point_angle, show_z = False,
                  title = "final pair")
    plt.show()
    '''
    #mpl_visualize(*final_gmm_pair, colors = ["g", "r"])
    for gmm_pair in merged_gmm_lists:
        mpl_visualize(*gmm_pair,
                      colors = ['y', 'g', 'r', 'b'], cov_scale = 2.0,
                      alpha = 0.2)
    '''
    #### compute scores
    # score the corrupted gmm with sampled mesh
    score_true = evaluation.eval_quality(true_gmm, true_pc)
    score_prior = evaluation.eval_quality(prior_gmm, true_pc)
    score_merged = evaluation.eval_quality(final_gmm, true_pc)

    print("Scores: true, prior, updated", score_true, score_prior, score_merged)
if __name__ == "__main__":
    main()
    print(" execution finished")
