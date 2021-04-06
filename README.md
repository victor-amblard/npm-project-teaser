This repository contains the code written as part of the final project for MVA's point cloud processing course ("Nuage de points et Modelisation 3D").


It is based on TEASER++ (paper: https://arxiv.org/abs/2001.07715, code: https://github.com/MIT-SPARK/TEASER-plusplus). It also uses feature detectors: FCGF (https://github.com/chrischoy/FCGF/tree/master/scripts) and 3DSmoothNet (https://github.com/zgojcic/3DSmoothNet).

To run the code, please make sure you have installed TEASER++ Python before (`import teaserpp_python` should then work) before (follow the instructions from the original Github repo).

Run `pip install -r requirements.txt` to install the additional modules you may need.

List of files in the repository:
* `util/` : utility files to process point clouds, poses, Kitti dataset, feature descriptors,...
* `generate_noise.py` : takes a point cloud and generates a noiy and transformed version of the point cloud with added outliers
* `scale_experiment.py` : tests for different values of voxel grid how TEASER++ behaves
* `stanford_dataset_tests.py` : a set of experiments made on the Bunny point cloud (changing outlier ratios, noise,...)
* `lidar_odometry.py` : small LIDAR experiment on Kitti dataset (requires to download sequence `2011_09_26_drive_0036`) http://www.cvlibs.net/datasets/kitti/raw_data.php 
