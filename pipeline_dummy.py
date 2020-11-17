import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from os.path import expanduser
import pickle
import scipy.linalg
import sklearn.mixture

home = expanduser("~")
print(home)
data_folder = home + "/semester_project/data"
bunny_mesh_file = data_folder + "/bunny/reconstruction/bun_zipper_res4.ply"
bunny_point_cloud_file = data_folder + "/bunny/data/bun045.ply" # create corrupted bunny
tmp_gmm_file = data_folder + "/tmp/tmp_gmm"

directGMM_folder = home + "/semester_project"

class gmm:
    def __init__(self, weights = [], means = [], covariances = []):
        self.weights = weights
        self.means = means
        self.covariances = covariances

def main():

    #required functionality
        # load measurement
        # (disrupt measurement)
    measurement_pc = load_bunny_measurement()
    measurement_gmm = simple_pc_gmm(measurement_pc, n= 100, recompute = False)

    # load mesh
        # (localize (rough) mesh location)
        # TODO: use directGMM()
    prior_mesh = load_bunny_mesh()
    prior_pc = sample_points(prior_mesh)

    # compute registration
        # various tools
        # possibilities: icp, gmm_reg, etc.
    transform = o3_point_to_point_icp(measurement_pc, prior_pc)

    #transform pc to the right spot
    measurement_registered = transform_measurement(measurement_pc, transform)

    # perform refinement
        #some magic stuff

    # visualize gmmm
    #o3_visualize(measurement_pc, prior_mesh, measurement_registered, mesh_sphere)
    mpl_visualize(measurement_pc, measurement_gmm, prior_mesh)


def load_bunny_measurement():
    return o3d.io.read_point_cloud(bunny_point_cloud_file)

def load_bunny_mesh():
    return o3d.io.read_triangle_mesh(bunny_mesh_file)

def o3_point_to_point_icp(source, target,threshold = 0.02, trans_init = np.identity(4)):
    # from: http://www.open3d.org/docs/0.9.0/tutorial/Basic/icp_registration.html
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    return reg_p2p.transformation

def transform_measurement(pc, transform):
    pc_tf = copy.deepcopy(pc)
    pc.paint_uniform_color([0.0, 0.0, 0.0])
    return pc_tf.transform(transform)

def sample_points(mesh, n_points = 10000):
    return mesh.sample_points_uniformly(n_points)

def simple_pc_gmm(pc, n = 100, recompute = True):
    if recompute:
        gmm_generator = sklearn.mixture.GaussianMixture(n_components = n)
        print("starting gmm_fit")
        gmm_generator.fit(pc.points)

        local_gmm = gmm()
        local_gmm.means = gmm_generator.means_
        local_gmm.weights = gmm_generator.weights_
        local_gmm.covariances = gmm_generator.covariances_

        pickle_out = open(tmp_gmm_file,"wb")
        pickle.dump(local_gmm, pickle_out)
        pickle_out.close()

    else:
        pickle_in = open(tmp_gmm_file,"rb")
        local_gmm = pickle.load(pickle_in)
        pickle_in.close()

    #print(gmm_generator.get_params())
    return local_gmm

def o3_visualize(*obj):
    #option: mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    o3d.visualization.draw_geometries(obj)

def mpl_visualize(*obj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sample_mesh = o3d.geometry.TriangleMesh()
    sample_pc = o3d.geometry.PointCloud()
    sample_gmm = gmm()

    for element in obj:
        if type(element) == type(sample_mesh):
            print("mesh detected")
            vertices = np.asarray(element.vertices)
            #print(points.shape)
            triangles = np.asarray(element.triangles)
            ax.plot_trisurf(vertices[:,0],vertices[:,1] , vertices[:,2],
                            triangles= triangles, linewidth=0.5, antialiased=True,
                            alpha = 0.4, label= "mesh", color = "b")

        elif type(element) == type(sample_pc):
            print("pc detected")
            points = np.asarray(element.points)

            max_point_nr = 1000
            step = max((len(points)//max_point_nr), 1)

            ax.scatter(points[1:-1:step,0], points[1:-1:step,1],
                       points[1:-1:step,2], c = 'r', s =0.5,
                       alpha = 0.7, label= "point cloud")

        elif type(element) == type(sample_gmm):
            print("gmm detected")
            centers = element.means
            ax.scatter(centers[1:-1,0], centers[1:-1,1],
                       centers[1:-1,2], c = 'g', s = 10.0, alpha = 0.7,
                       label= "gmm")
            for i in range(0,len(element.means)):
                local_cov = np.asarray(element.covariances[i])
                eigenvals, eigenvecs = scipy.linalg.eigh(local_cov)

                #ax.plot_surface(x,y,z)

        else:
            print("unkown type detected: " + type(element))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

if __name__ == "__main__":
    main()
