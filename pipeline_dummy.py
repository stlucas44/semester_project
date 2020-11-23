
import open3d as o3d
from os.path import expanduser

from lib.gmm_generation import gmm, simple_pc_gmm
from lib.registration import o3d_point_to_point_icp, transform_measurement
from lib.visualization import mpl_visualize, o3d_visualize

home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045.ply" # create corrupted bunny
tmp_gmm_file = data_folder + "/tmp/tmp_gmm"

directGMM_folder = home + "/semester_project"

def main():

    #required functionality
        # load measurement
        # (disrupt measurement)
    measurement_pc = load_bunny_measurement()
    measurement_gmm = simple_pc_gmm(measurement_pc, n= 100, recompute = False, path = tmp_gmm_file)

    # load mesh
        # (localize (rough) mesh location)
        # TODO: use directGMM()
    prior_mesh = load_bunny_mesh()
    prior_pc = sample_points(prior_mesh)

    # compute registration
        # various tools
        # possibilities: icp, gmm_reg, etc.
    transform = o3d_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    measurement_registered = transform_measurement(measurement_pc, transform)

    # perform refinement
        #some magic stuff

    # visualize gmmm
    o3d_visualize(measurement_pc, prior_mesh, measurement_registered)
    mpl_visualize(measurement_pc, prior_mesh, measurement_registered)
    mpl_visualize(measurement_gmm)

def load_bunny_measurement():
    return o3d.io.read_point_cloud(bunny_point_cloud_file)

def load_bunny_mesh():
    return o3d.io.read_triangle_mesh(bunny_mesh_file)

def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(n_points)

if __name__ == "__main__":
    main()
