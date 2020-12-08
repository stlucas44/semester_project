import copy
import numpy as np
import open3d as o3d

def o3d_point_to_point_icp(source, target,threshold = 1.0,
                          trans_init = np.identity(4)):
    # from: http://www.open3d.org/docs/0.9.0/tutorial/Basic/icp_registration.html
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    return reg_p2p.transformation

def transform_measurement(pc, transform):
    pc_tf = copy.deepcopy(pc)
    pc.paint_uniform_color([0.0, 0.0, 0.0])
    return pc_tf.transform(transform)
