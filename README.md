# Semester Project
Code base for the semster thesis **Local 3D Mesh Refinement via Gaussian Mixture Modelling**.

## Setup
To set this pipeline up you need to configure your virtual environment as written in `setup/requirements.txt` (TBA). Further
you need to build and install the repositories [directGMM](git@github.com:stlucas44/direct_gmm.git) (branch:`inital testing`) and [pymesh](https://pymesh.readthedocs.io/en/latest/index.html) (follow the install instructions on the site).


## Overview
All the building blocks of the pipeline are gathered in the `lib ` directory. The scripts on the top level represent various applications of the introduced tools.

### Scripts

* `automated_evaluation`: Scripted evaluation for a given set of parameters. `main` takes a parameters script and returns a box plot of the evaluation. The algorithm takes a mesh, corrupts it and then does the refinement step.

* `final_evaluation`: Operates `automated_evaluation` and feeds a set of parameters. Creates the plots shown in the reports evaluation.

* `gmm_comparison`: Intends to do analyze the t-test and shows the returned scores.
* `mesh_editor`: Helper script to edit and pertube meshes.
* `pipeline_dummy`: Example implementation of how to process a mesh and a pointcloud.
* `presentation_plots`: Same as `automated evaluation` but with fixed view points and sensor positions.
* `two_mesh_evaluation`: Mesh refinement for a ground true and a seperate corrupted mesh.
* `viusalize_mesh`: Helper script to select view points and camera positions.

### Pipeline components
All these libraries are found in `lib`
* `evaluation`: Contains all the evaluation score logic
* `gmm_generation`: All gmm transforms are implemented here.
* `loader`: Handles all interfaces to load meshes, pointclouds, gmms
* `merge`: Provides the T test with its helpers and a simple placeholder merge.
* `registration`: Finds the regirstation of the pointcloud with respect to  the given mesh. (Used in `pipeline_dummy`)
* `visualization`: Whole range of plotting facilities for meshes, pointclouds and gmms. Also creates match matrices and
  box plots.
