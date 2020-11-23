import sklearn
import pickle

class gmm:
    def __init__(self, weights = [], means = [], covariances = []):
        self.weights = weights
        self.means = means
        self.covariances = covariances

def simple_pc_gmm(pc, n = 100, recompute = True, path = None):
    if recompute:
        gmm_generator = sklearn.mixture.GaussianMixture(n_components = n)
        print("starting gmm_fit")
        gmm_generator.fit(pc.points)

        local_gmm = gmm()
        local_gmm.means = gmm_generator.means_
        local_gmm.weights = gmm_generator.weights_
        local_gmm.covariances = gmm_generator.covariances_

        pickle_out = open(path,"wb")
        pickle.dump(local_gmm, pickle_out)
        pickle_out.close()

    else:
        pickle_in = open(path,"rb")
        local_gmm = pickle.load(pickle_in)
        pickle_in.close()

    #print(gmm_generator.get_params())
    return local_gmm
