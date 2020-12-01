import copy
import open3d as o3d
from os.path import expanduser

from lib.gmm_generation import gmm
#from lib.registration import o3d_point_to_point_icp, transform_measurement
from lib import registration
from lib.visualization import *

import trimesh

home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045.ply"
cube_file = "test_cube.ply"
tmp_gmm_file = data_folder + "/tmp/tmp_gmm"

directGMM_folder = home + "/semester_project/direct_gmm/mixture"
hgmm_folder = home + "/semester_project/ \
                      GPU-Accelerated-Point-Cloud-Registration-Using-Hierarchical-GMM"

model_scaling = 10.0
# Note: Bunny is 150x50x120mm so factor 10 should work
cov_scale = 2.0 #95% quantile!

def main():

    # load measurement and (disrupt measurement)
    measurement_pc = load_measurement(bunny_point_cloud_file, model_scaling)

    #fit gmm
    measurement_gmm = gmm()
    measurement_gmm.pc_simple_gmm(measurement_pc, n = 50, recompute = False,
                                  path = tmp_gmm_file)
    measurement_gmm.sample_from_gmm()

    # load mesh (#TODO(stlucas): localize (rough) mesh location)
    prior_mesh = load_mesh(bunny_mesh_file, model_scaling)

    # fit via direct gmm
    prior_gmm = gmm()
    prior_gmm.mesh_gmm(prior_mesh, n = 100, recompute = True)

    prior_pc = sample_points(prior_mesh) # for final mesh evaluation

    # compute registration
        # various tools
        # possibilities: icp, gmm_reg, etc.
    #transform = registration.o3d_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    #measurement_registered = registration.transform_measurement(measurement_pc, transform)

    # perform refinement
        #some magic stuff

    # evaluate mesh
    ref_mesh = copy.deepcopy(prior_mesh)
    error_mesh = eval_quality(ref_mesh, prior_mesh)

    # visualize gmmm
    #o3d_visualize(measurement_pc, prior_mesh, measurement_registered)
    #mpl_visualize(measurement_pc, prior_mesh, measurement_registered)
    #mpl_visualize(measurement_pc, measurement_gmm, cov_scale = cov_scale)
    mpl_visualize(prior_gmm, cov_scale = cov_scale)

    #visualize_gmm_weights(measurement_gmm)

def load_measurement(path, scale):
    pc = o3d.io.read_point_cloud(bunny_point_cloud_file)
    return scale_o3d_object(pc, scale)

def load_mesh(path, scale):
    mesh = o3d.io.read_triangle_mesh(bunny_mesh_file)
    return scale_o3d_object(mesh, scale)

def scale_o3d_object(object, scale, scaling_center = np.zeros((3,1))):
    scaling_center = np.zeros((3,1))
    return object.scale(model_scaling, scaling_center)

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
