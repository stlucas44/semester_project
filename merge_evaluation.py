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
bunny_mesh_file_corrputed = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"

# params:
pc_sensor_position_enu = [0.0, 1.0, 2.0]
pc_sensor_rpy = [0.0, 90.0, 90.0]
pc_sensor_fov = [100, 85]
pc_range = 6.0
pc_angular_resolution = 0.3 # pico (354x287) would be 3.5 (res/fov)

n_pc_true = 1e9
print(n_pc_true)
n_pc_measurement = 1e6

#precomputed gmms:
tmp_gmm_true = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_true_pc = data_folder + "/tmp/tmp_measurement_gmm_pc"
tmp_gmm_prior = data_folder + "/tmp/tmp_mesh_gmm"


def main():
    #### true mesh
    # load true mesh
    true_mesh = load_mesh(bunny_mesh_file)
    # reduce to view point
    true_mesh_vp, true_mesh_occ = merge.view_point_crop(true_mesh,
           pc_sensor_position_enu,
           pc_sensor_rpy,
           sensor_max_range = pc_range,
           sensor_fov = pc_sensor_fov,
           angular_resolution = pc_angular_resolution)

    # sample it for comparison and update
    true_pc = sample_points(true_mesh_vp, n_points = 100) #n_pc_true)
    measurement_pc = sample_points(true_mesh_vp, n_points = n_pc_measurement)

    # generate gmms
    true_gmm = Gmm()
    true_gmm.mesh_gmm(true_mesh_vp, n = len(true_mesh.triangles), recompute = False, path = tmp_gmm_true)
    #true_gmm.naive_mesh_gmm(true_mesh_vp, mesh_std = 0.05)

    measurement_gmm = Gmm()
    measurement_gmm.mesh_gmm(true_mesh_vp, n = len(true_mesh.triangles), recompute = False, path = tmp_gmm_true)
    #measurement_gmm.naive_mesh_gmm(true_mesh_vp, mesh_std = 0.05)

    #### corrupted mesh
    # load corrupted mesh
    prior_mesh = load_mesh(bunny_mesh_file)
    # view point crop
    prior_mesh_vp, prior_mesh_occ = merge.view_point_crop(prior_mesh,
           pc_sensor_position_enu,
           pc_sensor_rpy,
           sensor_max_range = pc_range,
           sensor_fov = pc_sensor_fov,
           angular_resolution = pc_angular_resolution)

    # generate gmm
    prior_gmm = Gmm()
    prior_gmm.mesh_gmm(prior_mesh_vp, n = 300, recompute = False, path = tmp_gmm_prior)
    #prior_gmm.naive_mesh_gmm(prior_mesh_vp, mesh_std = 0.05)


    #### merge mesh with corrupted point cloud
    # apply merge with
    merged_gmm_lists, final_gmm, final_gmm_pair = merge.gmm_merge(
            prior_gmm,
            prior_gmm,
            p_crit = 0.05,
            sample_size = 5)

    #mpl_visualize(final_gmm)
    mpl_visualize(*final_gmm_pair, colors = ["g", "r"],
                  cov_scale = 2.0, show_mean = False,
                  view_angle = view_point_angle, show_z = False)
    #mpl_visualize(*final_gmm_pair, colors = ["g", "r"])


    for gmm_pair in merged_gmm_lists:
        mpl_visualize(*gmm_pair,
                      colors = ['y', 'g', 'r', 'b'], cov_scale = 2.0,
                      alpha = 0.2)


    #### compute scores
    # score the corrupted gmm with sampled mesh
    # score the updated gmm with the sampled original mesh



if __name__ == "__main__":
    main()
    print(" execution finished")
