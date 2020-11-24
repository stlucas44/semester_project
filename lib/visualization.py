import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import numpy as np
import open3d as o3d
import scipy

from lib.gmm_generation import gmm

def o3d_visualize(*obj):
    #option: mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    o3d.visualization.draw_geometries(obj)

def mpl_visualize(*obj, cov_scale = 1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for element in obj:
        if type(element) == type(o3d.geometry.TriangleMesh()):
            #print("mesh detected")
            visualize_mesh(element, ax)

        elif type(element) == type( o3d.geometry.PointCloud()):
            #print("pc detected")
            visualize_pc(element, ax)

        elif type(element) == type(gmm()):
            #print("gmm detected")
            visualize_gmm(element, ax, cov_scale = cov_scale)

        else:
            print("unkown type detected: " + type(element))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def visualize_mesh(mesh, ax, c = 'b', label = "mesh", alpha = 0.4, linewidth = 0.5):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    ax.plot_trisurf(vertices[:,0],vertices[:,1] , vertices[:,2],
                    triangles= triangles, linewidth=linewidth, antialiased=True,
                    alpha = alpha, label= label, color = c)
    return ax
def visualize_pc(pc, ax):
    points = np.asarray(pc.points)

    max_point_nr = 1000
    step = max((len(points)//max_point_nr), 1)

    ax.scatter(points[1:-1:step,0], points[1:-1:step,1],
               points[1:-1:step,2], c = 'r', s =0.5,
               alpha = 0.7, label= "point cloud")
    return ax

def visualize_gmm(gmm, ax, show_mean = True, cov_scale = 1.0):
    if show_mean:
        centers = gmm.means
        ax.scatter(centers[1:-1,0], centers[1:-1,1],
                   centers[1:-1,2], c = 'g', s = 10.0, alpha = 0.7,
                   label= "gmm")
    for i in range(0,len(gmm.means)):
        local_cov = np.asarray(gmm.covariances[i])
        local_mean = np.asarray(gmm.means[i])
        eigenvals, eigenvecs = scipy.linalg.eigh(local_cov)

        # quadric: (x-v)' * cov * (x-v) = 1
        # parametric representation
        # source: https://en.wikipedia.org/wiki/Ellipsoid#:~:text=An%20ellipsoid%20has%20three%20pairwise,simply%20axes%20of%20the%20ellipsoid.
        ellipse_resolution = 1.0
        phi = np.asarray([np.arange(0.0,2.0 * np.pi,ellipse_resolution)])
        theta = np.asarray([np.arange(-0.5 * np.pi, 0.5 * np.pi, 0.5 * ellipse_resolution)])

        # build range for parameters
        coscos = np.cos(theta.T) * np.cos(phi)
        cossin = np.cos(theta.T) * np.sin(phi)
        sin = np.sin(theta.T) * np.ones(shape = phi.shape)

        target_vector = np.reshape([coscos[:], cossin[:], sin[:]],
                                   (3, len(phi.T) * len(theta.T)))

        #print(target_vector.shape)
        local_mean_rep = np.dot(np.diag(local_mean), np.ones(shape=(3,len(phi.T)*len(theta.T))))

        #points = np.dot(np.diag([cov_scale, cov_scale, cov_scale]),
        #         np.dot(local_cov, target_vector)) + local_mean_rep

        cov_edited = np.dot(eigenvecs, np.diag(np.sqrt(eigenvals)))
        points = np.dot(np.diag([cov_scale, cov_scale, cov_scale]),
                 np.dot(cov_edited, target_vector)) + local_mean_rep

        ax.plot_trisurf(points[0,:],points[1,:], points[2,:],
                        linewidth=0.2, antialiased=True)
    return ax

def visualize_gmm_weights(gmm):
    fig = plt.figure()
    gmm_count = np.size(gmm.weights)
    #print(gmm.weights)
    plt.bar(np.arange(0,gmm_count), gmm.weights)
    plt.show()

def visualize_colored_samples(gmm):
    #print(np.size)
    for i in np.arange(0, len(gmm.samples)):
        visualize_pc
