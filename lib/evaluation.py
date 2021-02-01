import numpy as np
import open3d as o3d
import trimesh

from lib import loader
import matplotlib.pyplot as plt



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

def eval_quality(gmm, pc_true):
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


def eval_quality0(gmm, pc_true):
    proba = gmm.gmm_generator.predict_proba(np.asarray(pc_true.points))
    max_proba = proba.max(axis = 1)
    print(max_proba.shape)

    score = gmm.gmm_generator.score((np.asarray(pc_true.points)))
    return score
