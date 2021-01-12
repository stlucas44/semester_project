import copy
import numpy as np
import scipy
import sklearn.mixture
import sys
import pickle
import pymesh

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

            print("starting gmm_fit with num_points = ", self.num_train_points)

            self.gmm_generator.fit(pc.points)

            self.means = self.gmm_generator.means_
            self.weights = self.gmm_generator.weights_
            self.covariances = self.gmm_generator.covariances_
            if path is not None:
                pickle_out = open(path,"wb")
                pickle.dump(self.__dict__, pickle_out)
                pickle_out.close()
                print("wrote model to ", path)

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)

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

            print("finished hgmm fit")
            print("Resulted in : ", len(self.means), " mixtures")
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
            print("starting gmm_fit with vertices = ", self.num_train_points)

            #generate means, covs and faces
            com,a = get_centroids(mesh)
            print("Com ", com.shape, "      area (a) ", a.shape)
            face_vert = get_face_verts(mesh.vertices, mesh.faces)
            print("Face vert: ", face_vert.shape)

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

    def naive_mesh_gmm(self, init_mesh, mesh_std = 0.1):
        mesh = pymesh.meshio.form_mesh(np.asarray(init_mesh.vertices),
                                       np.asarray(init_mesh.triangles))

        means, aeras = get_centroids(mesh)
        face_vert = get_face_verts(mesh.vertices, mesh.faces)

        tri_covars = get_tri_covar(face_vert)
        #tri_covars = get_tri_covar_steiner(face_vert, means, mesh_std) -> not working yet!

        #guarantee covariances invertibility
        k = 0.001
        tri_covars = tri_covars + k * np.identity(3)

        self.means = means
        self.covariances = tri_covars
        self.num_gaussians = len(means)
        self.measured = np.zeros((self.num_gaussians,), dtype=bool)

        #generate precision and chol decomposition
        self.precs = np.zeros(self.covariances.shape)
        self.precs_chol = np.zeros(self.covariances.shape)
        counter = 0

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        for cov in self.covariances:
            if not is_pos_def(cov):
                print("negative semidefinite covariance matrix!")
                continue

            self.precs[counter, :, :] = np.linalg.inv(cov)
            self.precs_chol[counter, :, :] = np.linalg.inv(scipy.linalg.cholesky(cov))

        self.gmm_generator = sklearn.mixture.GaussianMixture(n_components = len(means), max_iter = 30)
        self.gmm_generator.means_ = self.means
        self.gmm_generator.weights_ = self.weights
        self.gmm_generator.covariances_ = self.covariances
        self.gmm_generator.precisions_ = self.precs
        self.gmm_generator.precisions_cholesky_ = self.precs_chol


    def sample_from_gmm(self, n_points = 1000):
        self.samples, self.sample_labels = self.gmm_generator.sample(n_points)

def extract_gmm(gmmA, maskA):
    merged_gmm = Gmm()

    merged_gmm.num_gaussians = len(gmmA.means[maskA])
    merged_gmm.means = np.asarray(gmmA.means[maskA])
    merged_gmm.weights = np.asarray(gmmA.weights[maskA])
    merged_gmm.covariances = np.asarray(gmmA.covariances[maskA])

    '''
    self.precs = np.asarray(self.precs)
    self.precs_chol = np.asarray(self.precs_chol)
    self.measured = np.ones((self.num_gaussians,), dtype=bool)


    merged_gmm.gmm_generator = sklearn.mixture.GaussianMixture(n_components = n_h, max_iter = 30)
    merged_gmm.gmm_generator.means_ = gmmA.means
    merged_gmm.gmm_generator.weights_ = gmmA.weights
    merged_gmm.gmm_generator.covariances_ = gmmA.covariances
    merged_gmm.gmm_generator.precisions_ = gmmA.precs
    merged_gmm.gmm_generator.precisions_cholesky_ = gmmA.precs_chol
    '''

    #TODO remove_dublicastes

    return merged_gmm


def merge_gmms(gmmA, maskA, gmmB, maskB):
    merged_gmm = Gmm()

    merged_gmm.num_gaussians = len(gmmA.means[maskA]) + len(gmmB.means[maskB])
    merged_gmm.means = np.concatenate([gmmA.means[maskA], gmmB.means[maskB]], axis = 0)
    merged_gmm.weights = np.concatenate([gmmA.weights[maskA], gmmB.weights[maskB]], axis = 0)
    merged_gmm.covariances = np.concatenate([gmmA.covariances[maskA], gmmB.covariances[maskB]], axis = 0)
    '''
    merged_gmm.precs = np.concatenate([gmmA.precs[maskA], gmmB.precs[maskB]], axis = 0)
    merged_gmm.precs_chol = np.concatenate([gmmA.precs_chol[maskA], gmmB.precs_chol[maskB]], axis = 0)
    merged_gmm.measured = np.concatenate([gmmA.measured[maskA], gmmB.measured[maskB]], axis = 0)


    merged_gmm.gmm_generator = sklearn.mixture.GaussianMixture(n_components = n_h, max_iter = 30)
    merged_gmm.gmm_generator.means_ = gmmA.means
    merged_gmm.gmm_generator.weights_ = gmmA.weights
    merged_gmm.gmm_generator.covariances_ = gmmA.covariances
    merged_gmm.gmm_generator.precisions_ = gmmA.precs
    merged_gmm.gmm_generator.precisions_cholesky_ = gmmA.precs_chol
    '''

    #TODO remove_dublicastes

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

def get_tri_covar_steiner(tris, centroids, mesh_std):
    print(tris.shape)
    # source: https://en.wikipedia.org/wiki/Steiner_ellipse

    AB = tris[:,1, :] - tris[:, 0, :]
    AC = tris[:,2, :] - tris[:, 0, :]
    BC = tris[:,2, :] - tris[:, 1, :]

    # determine SC as the longest avaliable! (actually does not )
    SC = tris[:,2,:] - centroids

    s_0_unbounded = SC
    l_0 = np.linalg.norm(s_0_unbounded, axis = 1).reshape((-1, 1))
    s_0 = s_0_unbounded / l_0
    #print(l_0.shape, s_0.shape)

    s_1_unbounded = 1/np.sqrt(3.0) * AB
    l_1 = np.linalg.norm(s_1_unbounded, axis = 1).reshape((-1, 1))
    s_1 = s_1_unbounded / l_1

    l_2 = np.square(mesh_std)
    s_2_unbounded = np.cross(AB, AC)
    s_2 = s_2_unbounded / np.linalg.norm(s_2_unbounded)

    cov = np.zeros(tris.shape)
    cov_unshaped = np.asarray([(l_0 * s_0), (l_1 * s_1), (l_2 * s_2)])
    print(cov_unshaped.shape)

    # TODO: make this transform work!
    range = np.arange(1, len(l_1))
    for i in range:
        cov[i] = cov_unshaped[:,i,:].T

    return cov
