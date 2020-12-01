import numpy as np
import sklearn.mixture
import sys
import pickle
import pymesh

sys.path.append('/home/lucas/semester_project/direct_gmm') # TODO(stlucas) find solution for this!
from mixture import GaussianMixture



class gmm:
    def __init__(self, weights = [], means = [], covariances = []):
        self.num_train_points = 0
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.gmm_generator = []
        self.samples = []
        self.sample_labels = []

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

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)

    def pc_hgmm(self, pc, n_h = 8, recompute = True, path = None):
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

            self.gmm_generator = GaussianMixture(n_components = n,
                                                 init_params=init_params,
                                                 max_iter=max_iter,tol=tol)

            self.num_train_points = len(mesh.vertices)
            print("starting gmm_fit with vertices = ", self.num_train_points)

            #generate means, covs and faces
            com,a = get_centroids(mesh)
            print("Com ", com.shape, "      a ", a.shape)
            face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1))
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
            print()

        else:
            if path is not None:
                pickle_in = open(path,"rb")
                tmp_dict = pickle.load(pickle_in)
                pickle_in.close()
                self.__dict__.update(tmp_dict)


    def sample_from_gmm(self, n_points = 1000):
        self.samples, self.sample_labels = self.gmm_generator.sample(n_points)


# direct gmm helper methods:
def get_centroids(mesh):
    # obtain a vertex for each face index
    face_vert = mesh.vertices[mesh.faces.reshape(-1),:].reshape((mesh.faces.shape[0],3,-1)) #@ np.array([[1,0,0],[0,0,1],[0,-1,0] ])
    # face_vert is size (faces,3(one for each vert), 3(one for each dimension))
    centroids = face_vert.sum(1)/3.0
    ABAC = face_vert[:,1:3,:] - face_vert[:,0:1,:]
    areas = np.linalg.norm(np.cross(ABAC[:,0,:],ABAC[:,1,:]),axis=1)/2.0
    #areas /= areas.min()
    #areas = areas.reshape((-1,1))
    return centroids, areas

def get_tri_covar(tris):
    covars = []
    for face in tris:
        A = face[0][:,None]
        B = face[1][:,None]
        C = face[2][:,None]
        M = (A+B+C)/3
        covars.append(A @ A.T + B @ B.T + C @ C.T - 3* M @ M.T)
    return np.array(covars)*(1/12.0)
