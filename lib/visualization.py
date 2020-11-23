import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import scipy

from lib.gmm_generation import gmm

def o3d_visualize(*obj):
    #option: mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    o3d.visualization.draw_geometries(obj)

def mpl_visualize(*obj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for element in obj:
        if type(element) == type(o3d.geometry.TriangleMesh()):
            print("mesh detected")
            vertices = np.asarray(element.vertices)
            #print(points.shape)
            triangles = np.asarray(element.triangles)
            ax.plot_trisurf(vertices[:,0],vertices[:,1] , vertices[:,2],
                            triangles= triangles, linewidth=0.5, antialiased=True,
                            alpha = 0.4, label= "mesh", color = "b")

        elif type(element) == type( o3d.geometry.PointCloud()):
            print("pc detected")
            points = np.asarray(element.points)

            max_point_nr = 1000
            step = max((len(points)//max_point_nr), 1)

            ax.scatter(points[1:-1:step,0], points[1:-1:step,1],
                       points[1:-1:step,2], c = 'r', s =0.5,
                       alpha = 0.7, label= "point cloud")

        elif type(element) == type(gmm()):
            print("gmm detected")
            show_mean = True
            if show_mean:
                centers = element.means
                ax.scatter(centers[1:-1,0], centers[1:-1,1],
                           centers[1:-1,2], c = 'g', s = 10.0, alpha = 0.7,
                           label= "gmm")
            for i in range(0,len(element.means)):
                local_cov = np.asarray(element.covariances[i])
                print(np.size(np.asarray(element.covariances)))
                local_mean = np.asarray(element.means[i])
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
                factor = 1000
                eigenvals = np.diag(factor * eigenvals)
                local_mean_spread = np.dot(np.diag(local_mean), np.ones(shape=(3,len(phi.T)*len(theta.T))))

                #points = np.dot(np.dot(eigenvecs, eigenvals), target_vector) + local_mean_spread
                points = np.dot(np.diag([factor, factor, factor]),
                         np.dot(local_cov, target_vector)) + local_mean_spread

                ax.plot_trisurf(points[0,:],points[1,:], points[2,:],
                                linewidth=0, antialiased=False)

        else:
            print("unkown type detected: " + type(element))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    #ax.legend()
    plt.show()
