import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors

from mpl_toolkits.mplot3d import Axes3D as ax
import numpy as np
import open3d as o3d
import scipy

from lib.gmm_generation import Gmm

# TODO: implement plt.draw() <-> show

def o3d_visualize(*obj):
    #option: mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    obj = [element.compute_vertex_normals() for element in obj]
    o3d.visualization.draw_geometries(obj)

def mpl_visualize(*obj, cov_scale = 1.0, colors = None, alpha = 0.4,
                  view_angle = None,
                  show_mean = True,
                  path = None,
                  show_z = True,
                  title = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = len(obj) * [None]
        print(colors)

    iterator = 0
    for element in obj:
        if type(element) == type(o3d.geometry.TriangleMesh()):
            if colors[iterator] is None:
                color = 'b'
            else:
                color = colors[iterator]
            #print("mesh detected")
            visualize_mesh(element, ax, c = color, alpha = alpha)

        elif type(element) == type( o3d.geometry.PointCloud()):
            if colors[iterator] is None:
                color = 'r'
            else:
                color = colors[iterator]
            #print("pc detected")
            visualize_pc(element, ax,  c = color)

        elif type(element) == type(Gmm()):
            #print("gmm detected")
            visualize_gmm(element, ax, cov_scale = cov_scale,
                          show_mean = show_mean, color = colors[iterator])

        else:
            print("unkown type detected: " + type(element))
        iterator = iterator + 1

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    if show_z:
        ax.set_zlabel('Z axis')
    else:
        ax.set_zticklabels([])

    if view_angle is not None:
        ax.view_init(*view_angle)

    if title is not None:
        ax.set_title(title)

    if path is not None:
        plt.savefig(path)

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

def visualize_pc(pc, ax = None, sensor_origin = None,
                     show = False, c = 'r'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    points = np.asarray(pc.points)

    max_point_nr = 1000
    samples = np.random.randint(0, len(points), (max_point_nr,))
    ax.scatter(points[samples,0], points[samples,1],
               points[samples,2], c = c, s =0.5,
               alpha = 0.7, label= "point cloud")
    if sensor_origin is not None:
        ax.scatter(sensor_origin[0], sensor_origin[1], sensor_origin[2], c = "b", s = 2.0,
        alpha = 0.7, label= "sensor_pos")

    if show:
        plt.show()

    return ax

def visualize_gmm(gmm, ax = None, show_mean = True, cov_scale = 1.0, show = False, color = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if show_mean:
        centers = gmm.means
        ax.scatter(centers[1:-1,0], centers[1:-1,1],
                   centers[1:-1,2], c = 'g', s = 5.0, alpha = 0.7,
                   label= "gmm")
    #colors = plt.cm.Pastel1(np.arange(0,gm_count)/(gm_count))

    gm_count = len(gmm.means)
    for i in range(0,gm_count):
        local_cov = np.asarray(gmm.covariances[i])
        local_mean = np.asarray(gmm.means[i])
        eigenvals, eigenvecs = scipy.linalg.eigh(local_cov)
        u,s,vt = np.linalg.svd(local_cov)

        # quadric: (x-v)' * cov * (x-v) = 1
        # parametric representation
        # source: https://en.wikipedia.org/wiki/Ellipsoid#:~:text=An%20ellipsoid%20has%20three%20pairwise,simply%20axes%20of%20the%20ellipsoid.

        rx, ry, rz = cov_scale*np.sqrt(s)#s#1/np.sqrt(coefs)

        R_reg = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T

        #print(eigs)
        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi/2.0, 10)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) #+ mean[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) #+ mean[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) #+ mean[2]

        for i in range(len(x)):
            for j in range(len(x)):
                x[i,j],y[i,j],z[i,j] = np.dot([x[i,j],y[i,j],z[i,j]], vt) \
                                       + local_mean
        # Plot:
        if color is not None:
            res = ax.plot_surface(x, y, z, shade=True, linewidth=0.0,
                                  color = color)
        else:
            res = ax.plot_surface(x, y, z, shade=True, linewidth=0.0)

        if show:
            plt.show()

    return ax

def visualize_gmm_weights(gmm):
    fig = plt.figure()
    gmm_count = np.size(gmm.weights)
    #print(gmm.weights)
    plt.bar(np.arange(0,gmm_count), gmm.weights)
    plt.show()

def visualize_match_matrix(match, score):
    # create discrete colormap
    cmap = mplcolors.ListedColormap(['red', 'green'])
    bounds = [0,0.5,1.0]
    norm = mplcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(match, cmap=cmap, norm=norm)
    ax[0].set_xlabel('measurement gmms (item numbers)')
    ax[0].set_ylabel('mesh gmms (item numbers)')
    ax[0].set_title('accepted t tests')

    intensity_cm = plt.get_cmap("autumn")
    print(" scaling the colormap to ", np.max(score))
    ax[1].imshow(score, cmap=intensity_cm, vmin= 0.0, vmax= np.max(score))
    ax[1].set_xlabel('measurement gmms (item numbers)')
    ax[1].set_ylabel('mesh gmms (item numbers)')
    ax[1].set_title("intersection heat map, p_max = " + str(np.max(score)))
    plt.show()
