# DROP: Dexterous Reorientation via Online Planning
This repository houses the code for the paper ["DROP: Dexterous Reorientation via Online Planning"](https://arxiv.org/abs/2409.14562). For a TL;DR, see the [paper website](https://caltech-amber.github.io/drop/). In particular, this repo contains the code that ran during our hardware experiments.

## Where to Find "X"
The implementations discussed in the paper are spread across a few different repositories in addition to this one. Here are some useful quick links to things that you might be curious about:
* The LEAP hand cube reorientation task in `mujoco_mpc` is located at [the `leap-hardware` branch of this fork](https://github.com/alberthli/mujoco_mpc/tree/leap-hardware).
    * We recommend building and running `mujoco_mpc` [using VSCode](https://github.com/alberthli/mujoco_mpc?tab=readme-ov-file#build-and-run-mjpc-gui-application-using-vscode), as suggested by the original authors.
    * All LEAP task-specific code is located [here](https://github.com/alberthli/mujoco_mpc/tree/leap-hardware/mjpc/tasks/leap).
    * For the simulated robustness trials, we need two separate task descriptions: one for the physics, and one for the planner's internal model of the physics. The physics task description is located [here](https://github.com/alberthli/mujoco_mpc/blob/leap-hardware/mjpc/tasks/leap/task.xml) and the planner's internal model is located [here](https://github.com/alberthli/mujoco_mpc/blob/leap-hardware/mjpc/tasks/leap/task_planner.xml). Right now, there's not much of a difference between these files. You can induce model error by messing with `task_planner.xml`, rebuilding, and re-running `mujoco_mpc`.
* The code for training the keypoint detector and custom smoother factors is located in [this repository](https://github.com/pculbertson/perseus). The main README explains how to both generate your own data as well as how to train/validate a model.
    * The weights for the RGBD and RGB models are located [here](/cube_rotation_ws/src/cro_ros/ckpts).
    * Augmentation details are [here](https://github.com/pculbertson/perseus/blob/main/perseus/detector/augmentations.py).
    * The keypoint detector model definition is [here](https://github.com/pculbertson/perseus/blob/main/perseus/detector/models.py).
    * The custom GTSAM smoother factors are [here](https://github.com/pculbertson/perseus/blob/main/perseus/smoother/factors.py).
* The code for the smoother logic is [here](/cro/perseus.py).
* The code for the corrector is [here](/cro/corrector.py). The function `timer_callback` is continuously called asynchronously in the estimator ROS2 node.
* The code for running `mujoco_mpc` in a ROS2 node is [here](/cube_rotation_ws/src/cro_control/src/cro_controller.cpp).
* The code for running the full estimator pipeline in a ROS2 node is [here](/cube_rotation_ws/src/cro_estimation/cro_estimation/cro_estimator.py) (the logic for querying the cameras in the hardware stack is located here as well).
* The various configurations for the hardware trials (including ablations) are located [here](/cube_rotation_ws/src/cro_ros/config/). The file [`hardware.yaml`](/cube_rotation_ws/src/cro_ros/config/hardware.yaml) is meant to be edited on-the-fly, and is the file that is run when booting up the stack. The other files are named in a way that is descriptive of their corresponding experiment. If you want to reproduce those trials, copy and paste their contents into `hardware.yaml` and see the [Usage section](#usage) below.

## Usage
If you have the hardware setup, you can run the stack by following the instructions in this section.

### Setup
The stack is run in a Docker container and uses `pixi` for the project-specific dependency management (dependencies from [Obelisk](https://github.com/Caltech-AMBER/obelisk), which is an AMBER Lab internal tool for managing hardware implementations for various platforms in the lab, are instead installed **locally in the Docker container**). The abbreviation `cro` throughout this repo stands for "cube rotation Obelisk," since we built the hardware code on top of Obelisk (you don't need to be familiar with this tool to understand the code).

First, clone the repo. We have large files managed by `git-lfs` in the form of checkpoints, so make sure you have git lfs installed.
```
# install and initialize git lfs
sudo apt install git-lfs
git lfs install

# clone this repo, large files should automatically be cloned properly
git clone https://github.com/Caltech-AMBER/drop.git
cd cube_rotation_obelisk

# initialize some custom rviz2 visualization plugins
git submodule update --init --recursive
```

Make sure you have Docker installed. Then, configure some Docker-related environment variables using our bash script one-liner and build the container (use `--dev` if you want development dependencies):
```
# run in repo root
bash setup.sh [--dev]
cd docker
docker compose run --build cube_rotation_obelisk
```

### Running the Stack
When in the container, you can boot up the stack:
```
# activate Obelisk features + build and source the cube rotation packages; configure the stack
# you only need to call setup one time unless the node crashes!
cro
setup

# in a separate terminal, start the stack when ready (see terminal messages)
cro
run

# temporarily stop the stack and re-home the hand
stop

# you can just type run again to start
run

# kill a run cleanly and release the hand motors
kill
```
While this is running, you can watch the visualizer with all rotation statistics in the `rviz` window that pops up.

## Citation
If you found either the code or the paper useful, please cite our work:
```
@article{li2024_drop,
    title={DROP: Dexterous Reorientation via Online Planning},
    author={Albert H. Li, Preston Culbertson, Vince Kurtz, and Aaron D. Ames},
    year={2024},
    journal={arXiv preprint arXiv:2409.14562},
    note={Available at: \url{https://arxiv.org/abs/2409.14562}},
}
```
