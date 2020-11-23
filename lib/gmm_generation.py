import sklearn.mixture
import pickle

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

    def sample(self, n_points = 1000):
        self.samples, self.sample_labels = self.gmm_generator.sample(n_points)
