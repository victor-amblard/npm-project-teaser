
import open3d as o3d
import copy 
import os 
import numpy as np
import time 
from open3d.open3d.geometry import voxel_down_sample, estimate_normals

from util_features import compute_fpfh_correspondences
from util_teaser import get_default_solver, transform_from_solution, get_angular_error
import generate_noise

def print_error(est_transf_se3, gt_transf_se3, est_scale=None, gt_scale = None):
    
    R_gt = gt_transf_se3[:3,:3]
    R_est = est_transf_se3[:3,:3]
    rot_err = get_angular_error(R_gt, R_est)

    t_gt = gt_transf_se3[:,-1]
    t_est = est_transf_se3[:,-1]
    tr_err = np.linalg.norm(t_gt - t_est)

    print("Translation error: {:.3f}m, Rotation error: {:.2f}deg".format(tr_err, rot_err))
    if est_scale is not None:
        scale_err = abs(est_scale - gt_scale) / gt_scale
        print(scale)
        print(gt_scale)
        print("Scale error: {:.3f}%".format(scale_err*100))
        return rot_err, tr_err, scale_err
    
    return rot_err, tr_err

def read_downsample(path, voxel_size):
    cloud = o3d.io.read_point_cloud(path)
    cloud = voxel_down_sample(cloud, voxel_size)

    return cloud 

def get_gt_transform(path):
    fn = os.path.join(os.path.dirname(path),"ground_truth_transform.txt")
    T = np.loadtxt(fn)
    scale = T[3,3]
    T[3,3] = 1
    return np.linalg.inv(T), 1/scale

VISUALIZATION = True

default_cloud_path = './data/model_bunny.ply'
transformed_cloud_path ='./data/transformed_bunny.ply'

T_gt, scale_gt = get_gt_transform(transformed_cloud_path)

all_voxel_sizes = [0.03, 0.05,0.07]
for voxel_size in all_voxel_sizes:  
    noise_bound = voxel_size

    default_cloud = read_downsample(default_cloud_path, voxel_size)
    transformed_cloud = read_downsample(transformed_cloud_path, voxel_size)
    
    if VISUALIZATION:
        o3d.visualization.draw_geometries([default_cloud, transformed_cloud])
    corr_a, corr_b = compute_fpfh_correspondences(default_cloud, transformed_cloud, voxel_size)

    start = time.time()
    teaser_solver = get_default_solver(noise_bound, estimate_scale=True)
    teaser_solver.solve(corr_b, corr_a)
    solution = teaser_solver.getSolution()
    end = time.time()
    T, scale = transform_from_solution(solution)
    print_error(T, T_gt, scale, scale_gt)

    if VISUALIZATION:
        cloud_copy = copy.deepcopy(transformed_cloud)
        cloud_copy.points = o3d.utility.Vector3dVector(np.asarray(transformed_cloud.points) * scale)
        cloud_copy.transform(T)
        cloud_copy.paint_uniform_color([0,0,1])
        default_cloud.paint_uniform_color([0,1,0])
        o3d.visualization.draw_geometries([default_cloud, cloud_copy])
