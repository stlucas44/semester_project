from automated_evaluation import *
from lib.loader import *
from lib.visualization import *
import gc

#paths:
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_file = data_folder + "/bunny/reconstruction/bun_zipper_res4_large.ply"
vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rhone_file =  data_folder + "/rhone_enu.off"
gorner_file = data_folder + "/gorner.off"
spiez_file = data_folder + "/mini_spiez_2/2_densification/3d_mesh/2020_09_17_spiez_simplified_3d_mesh.obj"


#settings:
bunny_mesh_params = {"path" : bunny_file, "aag" : (1.0, 3.0), "pc_sensor_fov" : [100, 85],
                     "disruption_range" : (0.0, 0.5),
                     "disruption_patch_size" : 0.15,
                     "refit_voxel_size" : 0.01,
                     "cov_condition" : 0.02,
                     "cov_condition_resampling" : 0.04,
                     "corruption_percentage" : 0.2,
                     "look_down" : False
                     }

curve_mesh_params = {"path" : curve_file, "aag" : (2.0,4.0), "pc_sensor_fov" : [80, 85],
                     "disruption_range" : (1.0, 2.0),
                     "disruption_patch_size" : 1.0,
                     "refit_voxel_size": 0.1,
                     "cov_condition" : 0.1,
                     "cov_condition_resampling" : 0.15,
                     "corruption_percentage" : 0.2,
                     "look_down" : True
                     }

vicon_params = {"path" : vicon_file, "aag" : (0.5, 2.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.05,
                "cov_condition_resampling" : 0.1,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }

gorner_params = {"path" : gorner_file, "aag" : (2.0, 4.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.1,
                "cov_condition" : 0.2,
                "cov_condition_resampling" : 0.6,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }

rhone_params = {"path" : rhone_file, "aag" : (0.5, 2.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.05,
                "cov_condition_resampling" : 0.1,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }

spiez_params = {"path" : spiez_file, "aag" : (0.5, 2.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.05,
                "cov_condition_resampling" : 0.1,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }


def eval_for_disruption():

    params_list = [bunny_mesh_params, curve_mesh_params]
    #params_list = [gorner_params]
    #params_list = [spiez_file]
    #params_list = [rohne_params]

    #params_list = [curve_mesh_params]


    corruptions = [0.05, 0.1, 0.2, 0.4]
    iterations_per_scale = 5
    results = np.zeros((len(corruptions), iterations_per_scale, 3))
    for params in params_list:
        for (scale_number, corruption_scale) in zip(np.arange(0,len(corruptions)),corruptions):
            for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
                print(("Starting on current scale: " + str(corruption_scale) +
                       " current iteration: " + str(iteration)).center(80,'*'))
                params["corruption_percentage"] = corruption_scale
                result = main(params) # result is [1,3]
                #print("worked again")
                results[scale_number, iteration] = result
                gc.collect()
                #print("results: ", results)
        labels = ["True", "Prior", "Refined"]
        draw_advanced_box_plots(results, labels, corruptions,
                                title = "Evaluation wrt prior quality (n = " + str(iterations_per_scale),
                                path = get_figure_path(params, "box"),
                                show = False)



        print(("finished with" + params['path']).center(100, '*'))
        gc.collect()


def eval_for_disruption_distance():
    params_list = [bunny_mesh_params, curve_mesh_params]
    params = bunny_mesh_params
    #params = curve_mesh_params
    #params = vicon_params

    distances = [0.0, 0.1, 0.5, 1.0, 2.0]

    iterations_per_scale = 5
    results = np.zeros((len(corruptions), iterations_per_scale, 3))
    for params in params_list:
        for (distance_number, distance) in zip(np.arange(0,len(distances)),distances):
            for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
                print(("Starting on current scale: " + str(corruption_scale) +
                       " current iteration: " + str(iteration)).center(80,'*'))
                params["disruption_range"] = (distances - 0.05, distances + 0.05)
                result = main(params) # result is [1,3]
                #print("worked again")
                results[distance_number, iteration] = result
                gc.collect()
                #print("results: ", results)
        labels = ["True", "Prior", "Refined"]
        draw_advanced_box_plots(results, labels, distances,
                                title = "Evaluation wrt prior quality (n = " + str(iterations_per_scale) + ")",
                                path = get_figure_path(params, "box"),
                                show = False,
                                xlabel = "avg batch offset ")



        print(("finished with" + params['path']).center(100, '*'))

    pass

def eval_for_smoothness():
    pass


if __name__ == "__main__":
    eval_for_disruption()
