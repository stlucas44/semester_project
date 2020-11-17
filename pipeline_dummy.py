import copy
import mesh_visualizer
import numpy as np
import open3d as o3d
from os.path import expanduser

home = expanduser("~")
print(home)
data_folder = home + "/semester_project/data"
bunny_mesh = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_point_cloud_mesh = data_folder + "/bunny/data/bun045.ply" # create corrupted bunny

directGMM_folder = home + "/semester_project"

def main():

    #required functionality
    # load measurement
    # (disrupt measurement)
    #measurement_mesh = load_bunny_measurement()
    measurement_pc = load_bunny_measurement()
    #measurement_pc = sample_points(measurement_mesh)

    # load mesh
        # (localize (rough) mesh location)
        # use directGMM()
    prior_mesh = load_bunny_mesh()
    prior_pc = sample_points(prior_mesh)

    # compute registration
        #various tools
        # possibilities: icp, gmm_reg, etc.
    transform = o3_point_to_point_icp(measurement_pc, prior_pc)
    measurement_registered = transform_measurement(measurement_pc, transform)
    # perform refinement
        #magic stuff

    # visualize gmmm
    visualize(measurement_pc, prior_mesh, measurement_registered)
    #mesh_visualizer.main()

    pass
def load_bunny_measurement():
    # TODO point cloud loading!
    # o3d.io.read_point_cloud(bunny_dummy)
    #return o3d.io.read_triangle_mesh(bunny_point_cloud_mesh)
    return o3d.io.read_point_cloud(bunny_point_cloud_mesh)

def load_bunny_mesh():
    mesh = o3d.io.read_triangle_mesh(bunny_mesh)
    return mesh

def o3_point_to_point_icp(source,
                          target,
                          threshold = 0.02,
                          trans_init = np.identity(4)):
    # from: http://www.open3d.org/docs/0.9.0/tutorial/Basic/icp_registration.html
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    #draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation

def transform_measurement(pc, transform):
    #print(type(pc))
    pc_tf = copy.deepcopy(pc)
    return pc_tf.transform(transform)

def sample_points(mesh, n_points = 10000):
    return o3d.geometry.sample_points_uniformly(mesh, n_points)

def visualize(*obj):
    o3d.visualization.draw_geometries(obj)

if __name__ == "__main__":
    main()
