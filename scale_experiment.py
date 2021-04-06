
import open3d as o3d
import copy 
import os 
import numpy as np
import time 
# from open3d.open3d.geometry import voxel_down_sample, estimate_normals
import subprocess
from util.util_features import compute_fpfh_correspondences
from util.util_teaser import get_default_solver, transform_from_solution, get_angular_error, print_error, certify_solution
import generate_noise


def read_downsample(path, voxel_size):
    cloud = o3d.io.read_point_cloud(path)
    cloud = cloud.voxel_down_sample(voxel_size)

    return cloud 

def get_gt_transform(path):
    fn = os.path.join(os.path.dirname(path),"ground_truth_transform.txt")
    T = np.loadtxt(fn)
    scale = T[3,3]
    T[3,3] = 1
    return np.linalg.inv(T), 1/scale

VISUALIZATION = True

default_cloud_path = './data/model_bunny.ply'

N_MC = 5
N_VOXELS = 4

results = np.zeros((N_MC, N_VOXELS, 4))

for i in range(N_MC):
    transformed_cloud_path ='./data/transformed_bunny.ply'
    args = "python  generate_noise.py " + default_cloud_path+ " " + transformed_cloud_path +  " --scale 0.5"
    subprocess.call(args, shell=True)

    T_gt, scale_gt = get_gt_transform(transformed_cloud_path)

    all_voxel_sizes = [0.01, 0.025, 0.05, 0.075,0.1]

    for j in range(N_VOXELS):
        voxel_size = all_voxel_sizes[j]  
        noise_bound = voxel_size

        default_cloud = read_downsample(default_cloud_path, voxel_size)
        transformed_cloud = read_downsample(transformed_cloud_path, voxel_size)
        
        if VISUALIZATION:
            transformed_cloud.paint_uniform_color([1,0,0])
            default_cloud.paint_uniform_color([0,0,1])
            o3d.visualization.draw_geometries([default_cloud, transformed_cloud])
        corr_a, corr_b = compute_fpfh_correspondences(default_cloud, transformed_cloud, voxel_size)

        start = time.time()
        teaser_solver = get_default_solver(noise_bound, estimate_scale=True)
        teaser_solver.solve(corr_b, corr_a)
        solution = teaser_solver.getSolution()
        # certify_solution(teaser_solver, solution.rotation, noise_bound)
        end = time.time()
        T, scale = transform_from_solution(solution)
        rot_err, tr_err, sc_err =  print_error(T, T_gt, scale, scale_gt)
        # Save results
        results[i, j, 0] = rot_err
        results[i, j, 1] = tr_err
        results[i, j, 2] = sc_err
        results[i, j, 3] = end - start 

        if VISUALIZATION:
            cloud_copy = copy.deepcopy(transformed_cloud)
            cloud_copy.points = o3d.utility.Vector3dVector(np.asarray(transformed_cloud.points) * scale)
            cloud_copy.transform(T)
            cloud_copy.paint_uniform_color([1,0,0])
            default_cloud.paint_uniform_color([0,0,1])
            o3d.visualization.draw_geometries([default_cloud, cloud_copy])

print(results)
print(np.mean(results, 0))
