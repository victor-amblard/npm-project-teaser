
import open3d as o3d
import copy 
import os 
import numpy as np
import time 
from open3d.open3d.geometry import voxel_down_sample, estimate_normals
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm 
import subprocess

from util.util_features import compute_fpfh_correspondences, extract_fpfh
from util.util_teaser import get_default_solver, transform_from_solution, get_angular_error, print_error
import generate_noise
from util.util_ransac import ransac_registration

def read_downsample(path, voxel_size):
    cloud = o3d.io.read_point_cloud(path)
    cloud = voxel_down_sample(cloud, voxel_size)

    return cloud 

def get_gt_transform(path):
    '''
        Reads a ground truth transformation (SE(3) + scale) saved by generate_noise.py
        Returns a numpy array corresponding to that transformation + scale
    '''
    fn = os.path.join(os.path.dirname(path),"ground_truth_transform.txt")
    T = np.loadtxt(fn)
    scale = T[3,3]
    T[3,3] = 1
    return np.linalg.inv(T), 1/scale

VISUALIZATION = False
default_cloud_path = './data/model_bunny.ply'

N_MC = 5 #Number of Monte Carlo runs
N_ALGOS = 2 # RANSAC + TEASER++
voxel_size = 0.05

param_grid = {'noise': [0.01, 0.02, 0.03], 'outliers' : [ 0.1, 0.25, 0.5, 0.75]}
grid = ParameterGrid(param_grid)
N_PARAMS = len(list(grid))

results = np.zeros((N_ALGOS, N_PARAMS, N_MC,4)) # 4 = rot_err + tr_err + sc_err + runtime 

for ii, elem in tqdm(enumerate(list(grid))):
    print()
    print("-------------------------")
    print("\t"+str(ii)+"\t")
    print("-------------------------")
    cur_noise_level = elem['noise']
    cur_outliers_level = elem['outliers']
    cur_overlap_level = 1 - cur_outliers_level #elem['partial_overlap']

    for j in range(N_MC):

        transformed_cloud_path ='./data/transformed_bunny.ply'
        args = "python  generate_noise.py " + default_cloud_path+ " " + transformed_cloud_path +  " --noise "+str(cur_noise_level) +\
            " --outliers " + str(cur_outliers_level) + " --partial_overlap "+str(cur_overlap_level)
        subprocess.call(args, shell=True)
        T_gt, scale_gt = get_gt_transform(transformed_cloud_path)

        noise_bound = 3 * cur_noise_level

        default_cloud = read_downsample(default_cloud_path, voxel_size)
        transformed_cloud = read_downsample(transformed_cloud_path, voxel_size)
        
        fpfh_feats_a, fpfh_raw_a = extract_fpfh(default_cloud, voxel_size)
        fpfh_feats_b, fpfh_raw_b= extract_fpfh(transformed_cloud, voxel_size)


        if VISUALIZATION:
            transformed_cloud.paint_uniform_color([1,0,0])
            default_cloud.paint_uniform_color([0,0,1])
            o3d.visualization.draw_geometries([default_cloud, transformed_cloud])
        
        corr_a, corr_b = compute_fpfh_correspondences(default_cloud, transformed_cloud, voxel_size)

        ### TEASER++
        start = time.time()
        teaser_solver = get_default_solver(noise_bound, estimate_scale=True)
        teaser_solver.solve(corr_b, corr_a)
        solution = teaser_solver.getSolution()
        end = time.time()
        T, scale = transform_from_solution(solution)
        
        rot_err, tr_err =  print_error(T, T_gt)
        # Save results
        results[0,ii, j, 0] = rot_err
        results[0,ii, j, 1] = tr_err
        results[0,ii, j, 2] = end - start 

        # ### RANSAC
        start = time.time()
        T = ransac_registration(default_cloud, transformed_cloud, fpfh_raw_a, fpfh_raw_b, voxel_size)
        end = time.time()

        rot_err, tr_err =  print_error(np.linalg.inv(T), T_gt)
        # Save results

        results[1,ii, j, 0] = rot_err
        results[1,ii, j, 1] = tr_err
        results[1,ii, j, 2] = end - start 

        if VISUALIZATION:
            cloud_copy = copy.deepcopy(transformed_cloud)
            cloud_copy.points = o3d.utility.Vector3dVector(np.asarray(transformed_cloud.points))
            cloud_copy.transform(T)
            cloud_copy.paint_uniform_color([1,0,0])
            default_cloud.paint_uniform_color([0,0,1])
            o3d.visualization.draw_geometries([default_cloud, cloud_copy])

print(results)
avg_results = np.mean(results, 2)
print(list(grid))
print("Results TEASER")
print(avg_results[0])

print("Results RANSAC")
print(avg_results[1])