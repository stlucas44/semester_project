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

def visualize_mesh(mesh, ax = None, c = 'b', label = "mesh", alpha = 0.4, linewidth = 0.5, show = False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    ax.plot_trisurf(vertices[:,0],vertices[:,1] , vertices[:,2],
                    triangles= triangles, linewidth=linewidth, antialiased=True,
                    alpha = alpha, label= label, color = c)
    if show:
        plt.show()

    return ax

def visualize_pc(pc, ax = None, show = False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    points = np.asarray(pc.points)

    max_point_nr = 1000
    step = max((len(points)//max_point_nr), 1)

    ax.scatter(points[1:-1:step,0], points[1:-1:step,1],
               points[1:-1:step,2], c = 'r', s =0.5,
               alpha = 0.7, label= "point cloud")
    return ax

def visualize_gmm(gmm, ax = None, show_mean = True, cov_scale = 1.0, show = False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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
        ellipse_resolution = 20
        phi = np.asarray([np.linspace(0.0,2.0 * np.pi,ellipse_resolution)])
        theta = np.asarray([np.linspace(-0.5 * np.pi, 0.5 * np.pi,int(ellipse_resolution/2))])

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
        points_centered = np.dot(np.diag([cov_scale, cov_scale, cov_scale]),
                 np.dot(cov_edited, target_vector))

        fancy = 2
        if fancy ==1:
            norm_mat =  np.asarray([np.cross(cov_edited[:,1], cov_edited[:,2]),
                                   np.cross(cov_edited[:,2], cov_edited[:,0]),
                                   np.cross(cov_edited[:,0], cov_edited[:,1])])
            normals = np.dot(norm_mat, target_vector)

            ellipse_pc = o3d.geometry.PointCloud()
            ellipse_pc.points = o3d.utility.Vector3dVector(points.T)
            ellipse_pc.normals = o3d.utility.Vector3dVector(normals.T)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                ellipse_pc, depth=9)

            print(np.asarray(mesh.triangles))
            ax.plot_trisurf(points[0,:],points[1,:], points[2,:],
                            triangles = np.asarray(mesh.triangles),
                            linewidth=0.2, antialiased=True)

        elif fancy ==2:
            #create convex hull with qHull
            #hull = scipy.spatial.ConvexHull(points_centered.T)
            hull = scipy.spatial.ConvexHull(points.T)

            ax.plot_trisurf(points[0,:],points[1,:], points[2,:],
                            triangles = hull.vertices,
                            linewidth=0.2, antialiased=True)

        else:
            ax.plot_trisurf(points[0,:],points[1,:], points[2,:],
                            linewidth=0.2, antialiased=True)

        if show:
            plt.show()

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
