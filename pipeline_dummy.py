import copy
import open3d as o3d
from os.path import expanduser

from lib import evaluation
from lib import registration
from lib import merge

from lib.gmm_generation import Gmm
from lib.loader import *
from lib.visualization import *

import trimesh

#define data path
home = expanduser("~")
data_folder = home + "/semester_project/src/data"
bunny_mesh_file = data_folder + "/bunny_corrupted.ply"
bunny_point_cloud_file = data_folder + "/bunny_pc.ply"

#loadable gmms for speed
tmp_gmm_measurement = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_mesh = data_folder + "/tmp/tmp_mesh_gmm"

# visualization
cov_scale = 2.0 #95% quantile!
view_point_angle =  (80.0, -60.0)

# tuning parameters:
num_gaussians_prior = 100
num_gaussians_measurement = 100
n_prior = 1.0# weights
n_measurement = 1.0

def main():
    ##### process measurement
    # load measurement and (disrupt measurement)
    measurement_pc = load_measurement(bunny_point_cloud_file)
    #TODO set (or get) cam pos and quat!
    sensor_position_enu = [0.0, 1.0, 2.0]
    sensor_rpy = [0.0, 90.0, 90.0]
    sensor_quaternion = []
    sensor_fov = [100, 85]
    range = 6.0
    angular_resolution = 0.3

    ##### process prior
    # load mesh (#TODO(stlucas): localize (rough) mesh location)
    prior_mesh = load_mesh(bunny_mesh_file)
    #prior_mesh = load_unit_mesh(type = "flat")
    view_point_mesh, occluded_mesh = view_point_crop(prior_mesh, sensor_position_enu,
                                   sensor_rpy, sensor_max_range = range,
                                   sensor_fov = sensor_fov,
                                   angular_resolution = angular_resolution)

    # fit via direct gmm
    prior_gmm = Gmm()
    prior_gmm.mesh_gmm(view_point_mesh, n = 300, recompute = False, path = tmp_gmm_mesh)
    #prior_gmm.naive_mesh_gmm(view_point_mesh, mesh_std = 0.05)

    view_point_mesh.compute_triangle_normals()
    view_point_mesh.compute_vertex_normals()
    mpl_visualize(prior_gmm , cov_scale = 2.0, colors = ['r', 'g'])
    mpl_visualize(prior_gmm, view_point_mesh , cov_scale = 2.0, colors = ['r', 'g'])

    ##### register and merge
    # compute registration
        # possibilities: icp, gmm_reg, etc.
    prior_pc = sample_points(prior_mesh, n_points = 10000) # for final mesh evaluation and registration
    transform = registration.o3d_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    measurement_registered = registration.transform_measurement(measurement_pc, transform)

    #fit measurement gmm
    measurement_gmm = Gmm()
    #measurement_gmm.pc_simple_gmm(measurement_registered, path = tmp_gmm_measurement, recompute = False)
    measurement_gmm.pc_hgmm(measurement_registered, path = tmp_gmm_measurement, recompute = False)

    # perform refinement
    #merged_pc = merge.simple_pc_gmm_merge(prior_pc, measurement_gmm)
    merged_gmm_lists, final_gmm, final_gmm_pair = merge.gmm_merge(prior_gmm, measurement_gmm, p_crit = 0.05, sample_size = 5)

    #mpl_visualize(final_gmm)
    mpl_visualize(*final_gmm_pair, colors = ["g", "r"],
                  cov_scale = 2.0, show_mean = False,
                  view_angle = view_point_angle, show_z = False)
    #mpl_visualize(*final_gmm_pair, colors = ["g", "r"])


    for gmm_pair in merged_gmm_lists:
        mpl_visualize(*gmm_pair,
                      colors = ['y', 'g', 'r', 'b'],
                      cov_scale = 2.0,
                      alpha = 0.2)

    ##### visualize

    #o3d_visualize(measurement_pc, prior_mesh, measurement_registered)
    mpl_visualize(measurement_pc, prior_mesh, measurement_registered, colors = ['y', 'r', 'b', 'g']) #registration
    #mpl_visualize(measurement_pc, measurement_gmm, cov_scale = cov_scale)# pc vs gmm
    #mpl_visualize(measurement_gmm, cov_scale = cov_scale)

    mpl_visualize(prior_gmm, cov_scale = cov_scale, colors = ['r'])
    #mpl_visualize(merged_pc, measurement_gmm, colors = ['r', 'g'],
    #              cov_scale = cov_scale)
    #mpl_visualize(measurement_pc)
    #visualize_pc(measurement_pc, sensor_origin = sensor_position_enu, show = True)

    #visualize distribution
    #visualize_gmm_weights(measurement_gmm)


if __name__ == "__main__":
    main()
