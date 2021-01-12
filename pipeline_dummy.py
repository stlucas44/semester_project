import copy
import open3d as o3d
from os.path import expanduser

from lib.gmm_generation import Gmm
#from lib.registration import o3d_point_to_point_icp, transform_measurement
from lib import registration
from lib.loader import *
from lib.visualization import *
from lib import merge

import trimesh

#define data path
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045_large.ply"
cube_file = "test_cube.ply"

#loadable gmms for speed
tmp_gmm_measurement = data_folder + "/tmp/tmp_measurement_gmm"
tmp_gmm_mesh = data_folder + "/tmp/tmp_mesh_gmm"

#path of local libs
directGMM_folder = home + "/semester_project/direct_gmm/mixture"
hgmm_folder = home + "/semester_project/ \
                      GPU-Accelerated-Point-Cloud-Registration-Using-Hierarchical-GMM"
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


    #fit measurement gmm
    measurement_gmm = Gmm()
    #options
    #measurement_gmm.pc_simple_gmm(measurement_pc, n = 50, recompute = False,
    #                              path = tmp_gmm_file)
    #measurement_gmm.pc_hgmm(measurement_pc, path = tmp_gmm_measurement, recompute = True)
    #measurement_gmm.sample_from_gmm()

    ##### process prior
    # load mesh (#TODO(stlucas): localize (rough) mesh location)
    prior_mesh = load_mesh(bunny_mesh_file)
    view_point_mesh, occluded_mesh = merge.view_point_crop(prior_mesh, sensor_position_enu,
                                   sensor_rpy, sensor_max_range = range,
                                   sensor_fov = sensor_fov,
                                   angular_resolution = angular_resolution)

    # fit via direct gmm
    prior_gmm = Gmm()
    prior_gmm.mesh_gmm(view_point_mesh, n = 200, recompute = False, path = tmp_gmm_mesh)
    #prior_gmm.naive_mesh_gmm(view_point_mesh)
    #prior_gmm.naive_mesh_gmm(prior_mesh)

    ##### register and merge
    # compute registration
        # possibilities: icp, gmm_reg, etc.
    prior_pc = sample_points(prior_mesh, n_points = 10000) # for final mesh evaluation and registration
    transform = registration.o3d_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    measurement_registered = registration.transform_measurement(measurement_pc, transform)
    #measurement_gmm.pc_simple_gmm(measurement_registered, path = tmp_gmm_measurement, recompute = False)
    measurement_gmm.pc_hgmm(measurement_registered, path = tmp_gmm_measurement, recompute = False)

    # perform refinement
    #merged_pc = merge.simple_pc_gmm_merge(prior_pc, measurement_gmm)
    merged_gmm_lists = merge.gmm_merge(prior_gmm, measurement_gmm)
    for gmm_pair in merged_gmm_lists:
        mpl_visualize(*gmm_pair, colors = ['r', 'b'], cov_scale = 2.0)

    # evaluate mesh
    #ref_mesh = copy.deepcopy(prior_mesh)
    #error_mesh = eval_quality(ref_mesh, prior_mesh)

    ##### visualize
    #presentation_plots:
    #presentation_plots(measurement_registered, prior_mesh, measurement_gmm, prior_gmm)

    #o3d_visualize(measurement_pc, prior_mesh, measurement_registered)
    mpl_visualize(measurement_pc, prior_mesh, measurement_registered, colors = ['r', 'b', 'g']) #registration
    #mpl_visualize(measurement_pc, measurement_gmm, cov_scale = cov_scale)# pc vs gmm
    #mpl_visualize(measurement_gmm, cov_scale = cov_scale)

    mpl_visualize(prior_gmm, cov_scale = cov_scale, colors = ['r'])
    #mpl_visualize(merged_pc, measurement_gmm, colors = ['r', 'g'],
    #              cov_scale = cov_scale)
    #mpl_visualize(measurement_pc)
    #visualize_pc(measurement_pc, sensor_origin = sensor_position_enu, show = True)

    #visualize distribution
    #visualize_gmm_weights(measurement_gmm)


# TODO(stlucas): to be moved to its own class?!
def presentation_plots(measurement_pc, prior_mesh, measurement_gmm, prior_gmm):
    mpl_visualize(measurement_pc, view_angle = view_point_angle, path="imgs/measurement_pc.png", show_z = False)
    mpl_visualize(prior_mesh, alpha = 1, view_angle = view_point_angle, path="imgs/prior_mesh.png", show_z = False)
    mpl_visualize(measurement_gmm, cov_scale = 2.0, show_mean = False, view_angle = view_point_angle,
                  path="imgs/measurement_gmm.png", show_z = False)
    mpl_visualize(prior_gmm, cov_scale = 2.0, show_mean = False, view_angle = view_point_angle,
                  path="imgs/prior_gmm.png", show_z = False)


def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(n_points)

def eval_quality(true_mesh, meas_mesh, num_points = 500):
    #pseudo shift
    vertices = np.asarray(meas_mesh.vertices)
    #vertices[:,1] = vertices[:,1] + 0.5

    faces = np.asarray(meas_mesh.triangles)

    eval_mesh = trimesh.Trimesh(vertices=vertices.tolist(),
                                faces = faces.tolist())
    true_pc = sample_points(true_mesh, num_points)
    (closest_points, distances, triangle_id) = \
        eval_mesh.nearest.on_surface(true_pc.points)

    #print("Eval results:\n", closest_points, distances, triangle_id)
    #print("Eval results:\n", type(distances))
    print("Eval results:\n", np.sort(distances)[:5])
    #print(np.mean(distances))

    #create new pc
    error_pc = o3d.geometry.PointCloud()
    sampled_points = np.asarray(true_pc.points)

    print(np.shape(sampled_points))
    sampled_points[:,2] = distances
    error_pc.points = o3d.utility.Vector3dVector(sampled_points)

    return error_pc

if __name__ == "__main__":
    main()
