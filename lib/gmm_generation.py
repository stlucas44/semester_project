import copy
import numpy as np
import scipy
import sklearn.mixture
import sys
import open3d as o3d
import pickle
import pymesh

import matplotlib.pyplot as plt

sys.path.append('/home/lucas/semester_project/direct_gmm') # TODO(stlucas) find solution for this!
from mixture import GaussianMixture

# defining small class for better handling
class HgmmHelper:
    def __init__(self, points, weight, mean = None, cov = None):
        self.points = points
        self.curr_weight = weight
        self.mean = mean
        self.var = cov

class Gmm:
    def __init__(self, weights = [], means = [], covariances = []):
        self.num_train_points = 0
        self.num_gaussians = 0
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.precs = []
        self.precs_chol = []
        self.gmm_generator = []
        self.samples = []
        self.sample_labels = []
        self.measured = False

    def pc_simple_gmm(self, pc, n = 100, recompute = True, path = None):
        # pc of type o3d.geometry.pointcloud
        if recompute:
            self.gmm_generator = sklearn.mixture.GaussianMixture(n_components = n)
            self.num_train_points = len(pc.points)

            print("  starting gmm_fit with num_points = ", self.num_train_points)

            self.gmm_generator.fit(pc.points)

            self.means = self.gmm_generator.means_
            self.weights = self.gmm_generator.weights_
            self.covariances = self.gmm_generator.covariances_
            self.precs = self.gmm_generator.precisions_
            self.precs_chol = self.gmm_generator.precisions_cholesky_
            if path is not None:
                pickle_out = open(path,"wb")
                pickle.dump(self.__dict__, pickle_out)
                pickle_out.close()
                print("  wrote model to ", path)

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)

        print("  finished gmm_fit")
    def pc_hgmm(self, pc, n_h = 8, recompute = True, path = None, min_points = 100):
        '''
        approach:
            object members:
                pointcollection
                overall weight
                mean
                var
        for each pointcloud do:
            fit gmm
            assign points to mixture components
            for each component:
                get cov ratio
                get mixture weight
                if cov_ratio > k or too few points in mixture:
                    add mixture component to final mixture
                else:
                    fill new object
                    new weight = overall weight * mixture weight
                    add object to new list

        '''
        self.num_train_points = len(pc.points)

        print("  starting hgmm_fit with num_points = ", self.num_train_points)

        all_points = np.asarray(pc.points)
        init_object = HgmmHelper(all_points, weight = 1.0)
        min_eig_ratio = 50
        max_third_cov = 0.01
        #stop_condition = lambda l1, l2, l3, num_points: (s[0]/s[2] > min_eig_ratio and
        #   s[1]/s[2] > min_eig_ratio) or num_points < min_points

        if recompute:
            # initialize list to iterate
            curr_list = [init_object]
            next_list = []

            #list of components to iterate through
            component_range = np.arange(0,n_h)

            while(len(curr_list) > 0):
                for sub_pc  in curr_list:
                    #fit local gmm
                    local_generator = sklearn.mixture.GaussianMixture(n_components = n_h, max_iter = 30)
                    labels = local_generator.fit_predict(sub_pc.points)
                    #print("labels: ", labels[1:10], "   local_generator.weights_ ", local_generator.weights_)

                    for i in component_range:
                        mean = local_generator.means_[i, :]
                        cov = local_generator.covariances_[i,:, :]
                        weight = local_generator.weights_[i]

                        #get member points of the mixture
                        local_indexes = [i for i, x in enumerate(labels == i) if x]
                        points = sub_pc.points[local_indexes, :]
                        #print("point shape: ", points.shape)

                        #get eigenvalue ratio
                        u, s, vt = np.linalg.svd(cov)

                        num_points = len(local_indexes)

                        #if (s[0]/s[2] > min_eig_ratio and
                        #   s[1]/s[2] > min_eig_ratio) or num_points < min_points:
                        if (2 * np.sqrt(s[2]) < max_third_cov or num_points < min_points):
                           #print(" sufficiently flat!")

                           self.means.append(mean)
                           self.weights.append(weight * sub_pc.curr_weight)
                           self.covariances.append(cov)
                           self.precs.append(local_generator.precisions_[i, :, :])
                           self.precs_chol.append(local_generator.precisions_cholesky_[i, : , :])

                           #print(np.asarray(self.means).shape)

                        else:
                           #print(" decompose this point cloud!")
                           new_pc = HgmmHelper(points, weight * sub_pc.curr_weight, mean = mean, cov = cov)
                           next_list.append(new_pc)

                        #print("finished iteration")
                curr_list = copy.deepcopy(next_list)
                next_list = []

            print("  finished hgmm, resulted in : ", len(self.means), " mixtures")
            self.num_gaussians = len(self.means)
            self.means = np.asarray(self.means)
            self.weights = np.asarray(self.weights)
            self.covariances = np.asarray(self.covariances)
            self.precs = np.asarray(self.precs)
            self.precs_chol = np.asarray(self.precs_chol)
            self.measured = np.ones((self.num_gaussians,), dtype=bool)

            self.gmm_generator = sklearn.mixture.GaussianMixture(n_components = n_h, max_iter = 30)
            self.gmm_generator.means_ = self.means
            self.gmm_generator.weights_ = self.weights
            self.gmm_generator.covariances_ = self.covariances
            self.gmm_generator.precisions_ = self.precs
            self.gmm_generator.precisions_cholesky_ = self.precs_chol

            if path is not None:
                pickle_out = open(path,"wb")
                pickle.dump(self.__dict__, pickle_out)
                pickle_out.close()

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)
        pass

    def mesh_gmm(self, init_mesh, n = 100, recompute = True, path = None, simple_fit = False):
        #transform o3d to pymesh
        mesh = pymesh.meshio.form_mesh(np.asarray(init_mesh.vertices),
                                       np.asarray(init_mesh.triangles))

        # transform mesh to pymesh
        if recompute:
            init_params = 'kmeans'
            tol = 1e-2
            max_iter = 100
            self.num_gaussians = n
            self.gmm_generator = GaussianMixture(n_components = n,
                                                 init_params=init_params,
                                                 max_iter=max_iter,tol=tol)

            self.num_train_points = len(mesh.vertices)
            print("  starting mesh_fit with vertices = ", self.num_train_points)

            #generate means, covs and faces
            com,a = get_centroids(mesh)
            print("  Com ", com.shape, "      area (a) ", a.shape)
            face_vert = get_face_verts(mesh.vertices, mesh.faces)
            print("  Face vert: ", face_vert.shape)

            data_covar = get_tri_covar(face_vert) # what does it expect?!

            #create gmm
            if simple_fit:
                self.gmm_generator.set_areas(a)
                self.gmm_generator.fit(com)
                self.gmm_generator.set_areas(None)

            else:
                self.gmm_generator.set_covars(data_covar)
                self.gmm_generator.set_areas(a)
                self.gmm_generator.fit(com)
                self.gmm_generator.set_covars(None)
                self.gmm_generator.set_areas(None)

            #write results
            self.means = self.gmm_generator.means_
            self.weights = self.gmm_generator.weights_
            self.covariances = self.gmm_generator.covariances_
            self.precs = self.gmm_generator.precisions_
            self.precs_chol = self.gmm_generator.precisions_cholesky_

            if path is not None:
                pickle_out = open(path,"wb")
                pickle.dump(self.__dict__, pickle_out)
                pickle_out.close()

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)

            else:
                print("  no path for loading mesh specified")

        print("  finished mesh_fit")


    def naive_mesh_gmm(self, init_mesh, mesh_std = 0.1):
        self.num_train_points = len(init_mesh.vertices)
        print("  starting naive_mesh_fit with vertices = ", self.num_train_points)

        mesh = pymesh.meshio.form_mesh(np.asarray(init_mesh.vertices),
                                       np.asarray(init_mesh.triangles))

        means, areas = get_centroids(mesh)
        face_vert = get_face_verts(mesh.vertices, mesh.faces)

        #tri_covars = get_tri_covar(face_vert)
        tri_covars = get_tri_covar_steiner(face_vert, means, areas, mesh_std)

        #guarantee covariances invertibility
        k = 0.001
        #tri_covars = tri_covars + k * np.identity(3)

        self.means = means
        self.covariances = tri_covars
        self.weights = areas/sum(areas)
        self.num_gaussians = len(means)
        self.measured = np.zeros((self.num_gaussians,), dtype=bool)

        #generate precision and chol decomposition
        self.precs = np.zeros(self.covariances.shape)
        self.precs_chol = np.zeros(self.covariances.shape)
        counter = 0

        def is_pos_def(x):
            U, S, V = np.linalg.svd(x)
            return np.all(S > 0)

        for cov in self.covariances:
            if not is_pos_def(cov):
                #print("negative semidefinite covariance matrix!")
                continue
            continue # DOTO fix the chol inversion!!
            self.precs[counter, :, :] = np.linalg.inv(cov)
            self.precs_chol[counter, :, :] = np.linalg.inv(scipy.linalg.cholesky(cov))

        self.gmm_generator = sklearn.mixture.GaussianMixture(n_components = len(means), max_iter = 30)
        self.gmm_generator.means_ = self.means
        self.gmm_generator.weights_ = self.weights
        self.gmm_generator.covariances_ = self.covariances
        self.gmm_generator.precisions_ = self.precs
        self.gmm_generator.precisions_cholesky_ = self.precs_chol
        print("  finished naive_mesh_fit ")


    def sample_from_gmm(self, n_points = 1000):
        self.samples, self.sample_labels = self.gmm_generator.sample(n_points)
        return_pc = o3d.geometry.PointCloud()
        return_pc.points = o3d.utility.Vector3dVector(self.samples)

        return return_pc


    def extract_gmm(self, maskA):
        merged_gmm = Gmm()

        weights_sum = self.weights[maskA].sum()

        merged_gmm.num_gaussians = len(self.means[maskA])
        merged_gmm.means = np.asarray(self.means[maskA])
        merged_gmm.weights = np.asarray(self.weights[maskA]) / weights_sum # crashing here!
        merged_gmm.covariances = np.asarray(self.covariances[maskA])

        merged_gmm.precs = np.asarray(self.precs[maskA])
        merged_gmm.precs_chol = np.asarray(self.precs_chol[maskA])
        merged_gmm.measured = np.ones((merged_gmm.num_gaussians,), dtype=bool)


        merged_gmm.gmm_generator = sklearn.mixture.GaussianMixture(
            n_components = len(merged_gmm.means), max_iter = 30)
        merged_gmm.gmm_generator.means_ = merged_gmm.means
        merged_gmm.gmm_generator.weights_ = merged_gmm.weights
        merged_gmm.gmm_generator.covariances_ = merged_gmm.covariances
        merged_gmm.gmm_generator.precisions_ = merged_gmm.precs
        merged_gmm.gmm_generator.precisions_cholesky_ = merged_gmm.precs_chol

        #TODO remove_dublicastes?
        return merged_gmm


def merge_gmms(gmmA, maskA, gmmB, maskB):
    merged_gmm = Gmm()

    weightsA = gmmA.weights[maskA].sum()
    weightsB = gmmB.weights[maskB].sum()
    weights_sum = weightsA + weightsB

    merged_gmm.num_gaussians = len(gmmA.means[maskA]) + len(gmmB.means[maskB])
    merged_gmm.means = np.concatenate([gmmA.means[maskA], gmmB.means[maskB]], axis = 0)
    merged_gmm.weights = np.concatenate([gmmA.weights[maskA], gmmB.weights[maskB]], axis = 0) / weights_sum
    merged_gmm.covariances = np.concatenate([gmmA.covariances[maskA], gmmB.covariances[maskB]], axis = 0)
    merged_gmm.precs = np.concatenate([gmmA.precs[maskA], gmmB.precs[maskB]], axis = 0)
    merged_gmm.precs_chol = np.concatenate([gmmA.precs_chol[maskA], gmmB.precs_chol[maskB]], axis = 0)
    #merged_gmm.measured = np.concatenate([gmmA.measured[maskA], gmmB.measured[maskB]], axis = 0)


    merged_gmm.gmm_generator = sklearn.mixture.GaussianMixture(n_components = merged_gmm.num_gaussians, max_iter = 30)
    merged_gmm.gmm_generator.means_ = merged_gmm.means
    merged_gmm.gmm_generator.weights_ = merged_gmm.weights
    merged_gmm.gmm_generator.covariances_ = merged_gmm.covariances
    merged_gmm.gmm_generator.precisions_ = merged_gmm.precs
    merged_gmm.gmm_generator.precisions_cholesky_ = merged_gmm.precs_chol

    #TODO check for dublicates
    return merged_gmm



# direct gmm helper methods: (from leonek)
def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = get_face_verts(mesh.vertices, mesh.faces)
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    #areas /= areas.min()
    #areas = areas.reshape((-1,1))
    return centroids, areas

def get_face_verts(vertices, faces):
    return vertices[faces.reshape(-1),:].reshape((faces.shape[0],3,-1))

def get_tri_covar(tris):
    # WARN: this algorithms creates non pos definite covariances!
    covars = []
    for face in tris:
        A = face[0][:,None]
        #print(" A = ", A)
        B = face[1][:,None]
        #print(" B = ", B)
        C = face[2][:,None]
        #print(" C = ", C)
        M = (A+B+C)/3
        covars.append(A @ A.T + B @ B.T + C @ C.T - 3* M @ M.T)
    return np.array(covars)*(1/12.0)

def get_tri_covar_steiner(tris, centroids, areas, mesh_std):
    cov = np.zeros(tris.shape)

    AB = tris[:,1, :] - tris[:, 0, :]
    AC = tris[:,2, :] - tris[:, 0, :]
    BC = tris[:,2, :] - tris[:, 1, :]
    SC = tris[:,2,:] - centroids

    for i in np.arange(0, len(AB)):
        ab = AB[i]
        sc = SC[i]

        # compute transform
        x = sc / np.linalg.norm(sc)
        z = np.cross(sc, ab)
        z = z / np.linalg.norm(z)
        y = np.cross(x, z)
        y = y / np.linalg.norm(y)

        transform = np.asarray([x, y, z]).T

        ab_l = np.matmul(transform.T, ab).round(9)
        sc_l = np.matmul(transform.T, sc).round(9)

        ab_retransf = np.matmul(transform, ab_l)
        sc_retransf = np.matmul(transform, sc_l)

        f1 = sc_l
        f2 = 1.0/np.sqrt(3) * ab_l

        upper = (np.square(f1).sum() - np.square(f2).sum())
        lower =  (2.0 * np.multiply(f1, f2).sum())
        cot_2t = np.divide(upper, lower)
        t_0 = np.arctan2(1, cot_2t) / 2

        #computing the half axis
        ha_0 = p_steiner(ab_l, sc_l, t_0).round(9)
        ha_1 = p_steiner(ab_l, sc_l, t_0 + (0.5 * np.pi)).round(9)

        #visualize_half_axis(transform, tris, ab_l, sc_l, i, centroids, t_0, ha_0, ha_1)

        local_area = np.pi * np.linalg.norm(ha_0) * np.linalg.norm(ha_1)

        #print("scaling factor", np.square(np.linalg.norm(ha_0)/2.0))
        scale_0 = np.square(np.linalg.norm(ha_0)/2.0).reshape((-1,))
        scale_1 = np.square(np.linalg.norm(ha_1)/2.0).reshape((-1,))

        first_axis = np.multiply(ha_0, scale_0)
        first_axis_scale = np.linalg.norm(first_axis)
        first_axis_unit = first_axis / first_axis_scale

        second_axis = np.multiply(ha_1, scale_1)
        second_axis_scale = np.linalg.norm(second_axis)
        second_axis_unit = second_axis / second_axis_scale

        third_axis_unit = np.cross(first_axis, second_axis)
        third_axis_unit = third_axis_unit / np.linalg.norm(third_axis_unit)
        third_axis_scale = np.square(mesh_std/2.0)

        #if any([first_axis_scale > 0.5, second_axis_scale  > 0.5]):
        #    visualize_half_axis(transform, tris, ab_l, sc_l, i, centroids, t_0, ha_0, ha_1)
        #    print(first_axis_scale, second_axis_scale, third_axis_scale)

        eigen_vals = np.diag([first_axis_scale, second_axis_scale, third_axis_scale])

        eigen_vecs = np.asarray([first_axis_unit, second_axis_unit, third_axis_unit]).T
        eigen_vecs = np.matmul(transform, eigen_vecs)

        cov[i,:,:] = np.linalg.multi_dot([eigen_vecs,eigen_vals,eigen_vecs.T])

        '''
        U, S, V = np.linalg.svd(local_cov)
        print("Eigs of local cov: ", S)

        U, S, V = np.linalg.svd(cov[i])
        print("Eigs of global cov: ", S)

        print("area ratio: ellipse, triangle: ", local_area / areas[i])
        area ratio is usually 2.418
        '''
    return cov

def p_steiner(AB, SC, t):
    return np.multiply(np.cos(t),SC)  + np.multiply (1/np.sqrt(3) * np.sin(t), AB)

def visualize_half_axis(transform, tris, ab_l, sc_l, i, centroids, t_0, ha_0, ha_1):
    a = np.matmul(transform.T, tris[i,0, :]).round(6)
    b = np.matmul(transform.T, tris[i,1, :]).round(6)
    c = np.matmul(transform.T, tris[i,2, :]).round(6)
    s = np.matmul(transform.T, centroids[i]).round(6)

    #print(" a, b, c: ", a, b, c)
    #print(" Local vectors:" , ab_l, sc_l)

    ha_2 = p_steiner(ab_l, sc_l, t_0 + (np.pi))
    ha_3 = p_steiner(ab_l, sc_l, t_0 - (0.5 * np.pi))
    ha = [ha_0, ha_1, ha_2, ha_3]

    #print("colinearity:", np.cross(ha_0 ,ha_1))
    plt.plot(a[0]-s[0], a[1]-s[1], "g.")
    plt.plot(b[0]-s[0], b[1]-s[1], "g.")
    plt.plot(c[0]-s[0], c[1]-s[1], "g.")
    plt.plot(s[0]-s[0], s[1]-s[1], "r.")

    for axis in ha:
        plt.plot([0.0, axis[0]], [0.0, axis[1]])
    plt.axis('equal')
    plt.show()

#from: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)
