import os 
import time 
import functools
import open3d as o3d
import numpy as np 
import copy 
from datetime import datetime
from open3d.open3d.geometry import voxel_down_sample, estimate_normals

from util_features import find_correspondences, extract_fpfh
from util_teaser import get_default_solver, transform_from_solution, certify_solution

import teaserpp_python


VOXEL_SIZE = 0.05
NOISE_BOUND = VOXEL_SIZE
NORMALIZATION_FACTOR = 25.

# To improve final registration
def icp(cloud_a, cloud_b, transf):
    icp_sol = o3d.registration.registration_icp(
            cloud_a, cloud_b, 0.05, transf,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))

    return icp_sol.transformation

def to_timestamp(fn1):
    '''
        Creates a timestamp from a lidar filename
    '''
    secs, nsecs = list(map(int, fn1.split(".")[0].split("_")[1:3]))
    dt = datetime.fromtimestamp(secs + float(nsecs) / 10**9)

    return dt 
def comp(fn1, fn2):
    '''
        Returns the chronological order for 2 LIDAR scans
    '''
    return to_timestamp(fn1) < to_timestamp(fn2)

def preprocess_cloud(folder, filename):
    '''
        Reads and downsamples a cloud using a voxel grid
    '''
    cloud_ply = o3d.io.read_point_cloud(os.path.join(folder, filename))
    cloud_ply.points = o3d.utility.Vector3dVector(np.asarray(cloud_ply.points) / NORMALIZATION_FACTOR)
    cloud_ply = voxel_down_sample(cloud_ply, VOXEL_SIZE)
    cloud = np.asarray(cloud_ply.points) 

    return cloud_ply, cloud

def load_all_filenames(filename):
    '''
        Loads all LIDAR scans from a file
    '''
    all_filenames = []
    with open(filename, 'r') as f:
        all_lines = list(map(str.strip, f.readlines()))
        for line in all_lines[1:]:
            scan = line.split()[-1][:-1]
            if not scan.endswith(".pcd"):
                continue
            # all_filenames.append(os.path.basename(scan))
            all_filenames.append(scan)
            folder = os.path.dirname(filename)
    
    all_filenames = sorted(all_filenames, key=functools.cmp_to_key(comp))

    return all_filenames, folder

def visualization(cloud_1, cloud_2, transf):
    A_pcd_T_teaser = copy.deepcopy(cloud_1).transform(transf)
    
    cloud_2.paint_uniform_color([1,0,0])
    A_pcd_T_teaser.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([A_pcd_T_teaser, cloud_2])

def get_closest_cloud(now, ptr, all_files):
    '''
        Returns the index of the closest cloud and a boolean (success)
    '''

    while ptr < len(all_files) and to_timestamp(all_files[ptr]) < now:
        ptr += 1

    if ptr == len(all_files):
        return -1, False

    else:
        return ptr - 1, True

if __name__ == "__main__":
    # fn = "/home/victor/Data/Stages/MIT/newer_college_dataset/seq4/0/filenames.txt"
    fn = "/home/victor/Data/Stages/MIT/newer_college_dataset/ouster_scan/filenames.txt"

    all_files, folder =  load_all_filenames(fn)

    prev_cloud, prev_cloud_npy = preprocess_cloud(folder, all_files[0])
    prev_feats = extract_fpfh(prev_cloud,VOXEL_SIZE)
    
    initial_transform = np.eye(4)
    origin = np.zeros(4)
    origin[-1] = 1

    positions = np.zeros((len(all_files), 4))
    positions[0] = origin

    initial_time = to_timestamp(all_files[0])
    start = datetime.now()

    idx = 0
    idx_max = all_files.index("cloud_1583840499_935335936.pcd") # Last point cloud for this experiment 
    
    i = 0

    while True:
        print("Now processing "+all_files[idx])
        teaser_solver = get_default_solver(NOISE_BOUND, estimate_scale = False)

        cur_cloud, cur_cloud_npy = preprocess_cloud(folder, all_files[idx])
        cur_feats = extract_fpfh(cur_cloud,VOXEL_SIZE)

        A_corr, B_corr = find_correspondences(prev_feats, cur_feats)

        # o3d.visualization.draw_geometries([cur_cloud])
        teaser_solver.solve(prev_cloud_npy[A_corr].T, cur_cloud_npy[B_corr].T)
        solution = teaser_solver.getSolution()
        transformation, scale = transform_from_solution(solution)
        icp_transf = icp(prev_cloud, cur_cloud, transformation)
        
        # certify_solution(teaser_solver, solution.rotation, NOISE_BOUND)
        # visualization(prev_cloud, cur_cloud, icp_transf)

        positions[i] = icp_transf @ positions[i-1]
        prev_cloud = cur_cloud
        prev_cloud_npy = cur_cloud_npy.copy()
        prev_feats = cur_feats.copy()
        
        delta_t = datetime.now() - start
        now_tmstamp = initial_time + delta_t
        idx, success = get_closest_cloud(now_tmstamp, idx, all_files)

        if not success or idx == idx_max:
            break 
        
        i += 1

    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(positions[:,:3])
    gt_seq = o3d.io.read_point_cloud("/home/victor/Data/Stages/MIT/newer_college_dataset/seq4/gt_seq4.pcd")
    new_pts = np.asarray(gt_seq.points)
    new_pts -= new_pts[0]
    gt_seq.points = o3d.utility.Vector3dVector(new_pts / NORMALIZATION_FACTOR)    
    gt_seq.paint_uniform_color([0,0,1])
    pcloud.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pcloud, gt_seq])