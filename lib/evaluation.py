import numpy as np
import open3d as o3d
import trimesh

from lib import loader
import matplotlib.pyplot as plt


def boundary_likelihood(sigma_range = 2.0):
    mean = np.asarray([0.0 ,0.0, 0.0])
    cov = np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

    interesting_points = np.asarray([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])

    likelihoods = multivariate_normal.pdf(interesting_points,mean, mean)
    print(likelihoods)

    return likelihoods

def maha_dist(x, mean, prec):
    delta = (x - mean).reshape(3,1)

    prod = np.linalg.multi_dot([delta.T, prec, delta])
    return np.sqrt(prod)

def eval_quality_mesh(true_mesh, meas_mesh, num_points = 500):
    #pseudo shift
    vertices = np.asarray(meas_mesh.vertices)
    #vertices[:,1] = vertices[:,1] + 0.5

    faces = np.asarray(meas_mesh.triangles)

    eval_mesh = trimesh.Trimesh(vertices=vertices.tolist(),
                                faces = faces.tolist())
    true_pc = loader.sample_points(true_mesh, num_points)
    (closest_points, distances, triangle_id) = \
        eval_mesh.nearest.on_surface(true_pc.points)

    #print("Eval results:\n", closest_points, distances, triangle_id)
    #print("Eval results:\n", type(distances))
    print("Eval results:\n", np.sort(distances)[:5])
    #print(np.mean(distances))

    #create new pc
    error_pc = o3d.geometry.PointCloud()
    sampled_points = np.asarray(true_pc.points)

    print(np.shape(sampled_points))
    sampled_points[:,2] = distances
    error_pc.points = o3d.utility.Vector3dVector(sampled_points)

    return error_pc

def eval_quality_proba(gmm, pc_true):
    proba = gmm.gmm_generator.predict_proba(np.asarray(pc_true.points))
    print("probabilities shape: ", proba.shape)
    sum_proba = np.sum(proba, axis = 1)
    mean_proba = np.mean(proba, axis = 1)

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

    axs[0].hist(proba)
    axs[1].hist(sum_proba)
    axs[2].hist(mean_proba)

    plt.show()
    max_proba = proba.max(axis = 1)
    print(max_proba.shape)

    score = gmm.gmm_generator.score((np.asarray(pc_true.points)))
    return score

def eval_quality_maha(gmm, pc_true):
    points = np.asarray(pc_true.points)
    predictions = gmm.gmm_generator.predict(points)
    maha_list = np.zeros(predictions.shape)

    means = gmm.means
    precs = gmm.precs

    iterator = 0
    for (point, prediction, maha) in zip(points, predictions, maha_list):
        maha_list[iterator] = maha_dist(point, means[prediction], precs[prediction])
        iterator = iterator + 1

    score = float(sum(maha_list < 2.0)) / len(maha_list)
    return score

def eval_quality(gmm, pc_true, type = 'maha'):
    if type == 'maha':
        return eval_quality_maha(gmm, pc_true)
    elif type == 'score':
        return eval_quality_score(gmm, pc_true)
    elif type == 'proba':
        return eval_quality_score(gmm, pc_true)
    else:
        print("Unkown score!")
        return 0.0

def eval_quality_score(gmm, pc_true):
    score = gmm.gmm_generator.score((np.asarray(pc_true.points)))
    return score
