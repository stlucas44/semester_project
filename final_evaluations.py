from automated_evaluation import *
from lib.loader import *
from lib.visualization import *
import gc

#paths:
home = expanduser("~")
data_folder = home + "/semester_project/data"
bunny_file = data_folder + "/bunny.ply"
vicon_file = data_folder + "/vicon.stl"
curve_file = data_folder + "/curve.off"
rhone_file =  data_folder + "/rhone_enu_reduced.off"
huenli_file = data_folder + "/gorner.off"
spiez_file = data_folder + "/spiez_reduced.obj"


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
                     "disruption_range" : (0.5, 2.0),
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

huenli_params = {"path" : huenli_file, "aag" : (2.0, 4.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.1,
                "cov_condition" : 0.2,
                "cov_condition_resampling" : 0.6,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }

rhone_params = {"path" : rhone_file, "aag" : (40.0, 60.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 5.0),
                "disruption_patch_size" : 0.5,
                "refit_voxel_size": 0.1,
                "cov_condition" : 0.5,
                "cov_condition_resampling" : 1.0,
                "corruption_percentage" : 0.2,
                "look_down" : True
                }

spiez_params = {"path" : spiez_file, "aag" : (0.5, 2.0), "pc_sensor_fov" : [100, 85],
                "disruption_range" : (0.5, 2.0),
                "disruption_patch_size" : 0.8,
                "refit_voxel_size": 0.05,
                "cov_condition" : 0.2,
                "cov_condition_resampling" : 0.3,
                "corruption_percentage" : 0.2,
                "look_down" : False
                }

aic = True

def eval_for_disruption():
    #params_list = [bunny_mesh_params, curve_mesh_params, spiez_params, rhone_params]

    corruptions = [0.05, 0.1, 0.2, 0.4]
    iterations_per_scale = 10
    results = np.zeros((len(corruptions), iterations_per_scale, 3))
    if aic:
        aic_results = np.zeros((len(corruptions), iterations_per_scale, 3))
    for params in params_list:
        for (scale_number, corruption_scale) in zip(np.arange(0,len(corruptions)),corruptions):
            for (iteration, result) in zip(np.arange(0,iterations_per_scale), results):
                print(("Starting on current scale: " + str(corruption_scale) +
                       " current iteration: " + str(iteration)).center(80,'*'))
                params["corruption_percentage"] = corruption_scale
                if aic:
                    results[scale_number, iteration], aic_results[scale_number, iteration] = main(params, aic = aic)
                else:
                    results[scale_number, iteration] = main(params, aic = aic) # result is [1,3]

                #print("worked again")
                gc.collect()
                #print("results: ", results)
        labels = ["True", "Prior", "Refined"]
        draw_advanced_box_plots(results, labels, corruptions,
                                title = get_name(params['path']) +
                                "\n Maha Evaluation wrt prior mesh quality (n = " + str(iterations_per_scale) + ")",
                                path = get_figure_path(params, "box"),
                                show = False)

        if aic:
            draw_advanced_box_plots(np.log(aic_results), labels, corruptions,
                                    title = get_name(params['path']) +
                                    "\n AIC Evaluation wrt prior quality (n = " + str(iterations_per_scale) + ")",
                                    path = get_figure_path(params, "aic_box"),
                                    show = False,
                                    ylabel = "log P"
                                    )
        print(("finished with" + params['path']).center(100, '*'))
        gc.collect()


if __name__ == "__main__":
    eval_for_disruption()
