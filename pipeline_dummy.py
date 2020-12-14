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

home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large_corrupted.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045_large.ply"
cube_file = "test_cube.ply"
tmp_gmm_file = data_folder + "/tmp/tmp_measurement_gmm"

directGMM_folder = home + "/semester_project/direct_gmm/mixture"
hgmm_folder = home + "/semester_project/ \
                      GPU-Accelerated-Point-Cloud-Registration-Using-Hierarchical-GMM"

model_scaling = 10.0
# Note: Bunny is 150x50x120mm so factor 10 should work
cov_scale = 2.0 #95% quantile!

def main():
    ##### process measurement
    # load measurement and (disrupt measurement)
    measurement_pc = load_measurement(bunny_point_cloud_file)
    #TODO set (or get) cam pos and quat!
    measurement_origin = [0.0, 1.0, 3.0]

    #fit gmm
    measurement_gmm = Gmm()
    #measurement_gmm.pc_simple_gmm(measurement_pc, n = 50, recompute = False,
    #                              path = tmp_gmm_file)
    #measurement_gmm.pc_hgmm(measurement_pc)
    #measurement_gmm.sample_from_gmm()

    ##### process prior
    # load mesh (#TODO(stlucas): localize (rough) mesh location)
    prior_mesh = load_mesh(bunny_mesh_file)

    # fit via direct gmm
    #prior_gmm = Gmm()
    #prior_gmm.mesh_gmm(prior_mesh, n = 100, recompute = True)
    prior_pc = sample_points(prior_mesh, n_points = 10000) # for final mesh evaluation

    ##### register and merge
    # compute registration
        # possibilities: icp, gmm_reg, etc.
    transform = registration.o3d_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    measurement_registered = registration.transform_measurement(measurement_pc, transform)
    measurement_gmm.pc_hgmm(measurement_registered, path = tmp_gmm_file, recompute = False)

    # perform refinement
    #merged_pc = merge.simple_pc_gmm_merge(prior_pc, measurement_gmm)

    # evaluate mesh
    #ref_mesh = copy.deepcopy(prior_mesh)
    #error_mesh = eval_quality(ref_mesh, prior_mesh)

    ##### visualize
    #o3d_visualize(measurement_pc, prior_mesh, measurement_registered)
    #mpl_visualize(measurement_pc, prior_mesh, measurement_registered, colors = ['r', 'b', 'g']) #registration
    #mpl_visualize(measurement_pc, measurement_gmm, cov_scale = cov_scale)# pc vs gmm
    #mpl_visualize(measurement_gmm, cov_scale = cov_scale)
    #mpl_visualize(prior_gmm, cov_scale = cov_scale)
    #mpl_visualize(merged_pc, measurement_gmm, colors = ['r', 'g'],
    #              cov_scale = cov_scale)
    mpl_visualize(measurement_pc)
    visualize_pc(measurement_pc, sensor_origin = measurement_origin, show = True)

    #visualize distribution
    #visualize_gmm_weights(measurement_gmm)

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
