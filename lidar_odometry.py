import os 
import time 
import functools
import open3d as o3d
import numpy as np 
import copy 
from datetime import datetime
from open3d.open3d.geometry import voxel_down_sample, estimate_normals

from util.util_features import find_correspondences, extract_fpfh
from util.util_teaser import get_default_solver, transform_from_solution, certify_solution
import util.util_kitti 
import matplotlib.pyplot as plt

import teaserpp_python


VOXEL_SIZE = 0.025
NOISE_BOUND = 0.01#VOXEL_SIZE
NORMALIZATION_FACTOR = 150.



def read_kitti(folder, filename):
    '''
        Reads a Kitti like velodyne point (.bin)
    '''
    pointcloud = np.fromfile(os.path.join(folder, filename), dtype=np.float32, count=-1).reshape([-1,4])
 
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    
    cloud_ply_points = np.vstack((x,y,z)).T
    
    cloud_ply = o3d.geometry.PointCloud()
    cloud_ply.points = o3d.utility.Vector3dVector(np.asarray(cloud_ply_points) / NORMALIZATION_FACTOR)
    cloud_ply = voxel_down_sample(cloud_ply, VOXEL_SIZE)
    cloud = np.asarray(cloud_ply.points) 

    return cloud_ply, cloud

def kitti_read_timestamps(folder):
    '''
        Reads Kitti timestamp
    '''
    allTs = []
    path = os.path.join(folder, "../timestamps.txt")
    with open(path, 'r') as f:
        allLines = list(map(str.strip, f.readlines()))
        for line in allLines:
            curtm = datetime.strptime(line[:-3], '%Y-%m-%d %H:%M:%S.%f') #Discarding ns
            allTs.append(curtm)
    return allTs

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


def load_all_filenames(filename):
    '''
        Loads all LIDAR scans from a file
    '''
    all_filenames = []
    folder = None
    with open(filename, 'r') as f:
        all_lines = list(map(str.strip, f.readlines()))
        for line in all_lines[1:]:
            scan = line.split()[-1]
            print(scan)
            if not scan.endswith(".pcd") and not scan.endswith(".bin"):
                continue
            # all_filenames.append(os.path.basename(scan))
            all_filenames.append(scan)
            folder = os.path.dirname(filename)
    
    # all_filenames = sorted(all_filenames, key=functools.cmp_to_key(comp))

    return all_filenames, folder

def visualization(cloud_1, cloud_2, transf):
    '''
        Visualization helper
    '''
    A_pcd_T_teaser = copy.deepcopy(cloud_1).transform(transf)
    
    cloud_2.paint_uniform_color([1,0,0])
    A_pcd_T_teaser.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([A_pcd_T_teaser, cloud_2])

def get_closest_cloud(now, ptr, timestamps):
    '''
        Returns the index of the closest cloud and a boolean (success)
    '''

    while ptr < len(timestamps) and timestamps[ptr] < now:
        ptr += 1

    if ptr == len(timestamps):
        return -1, False

    else:
        return ptr - 1, True

def load_gt_imu_frame(kitti_base_dir, all_files):
    ground_truth_poses = util_kitti.load_oxts_packets_and_poses([os.path.join(os.path.dirname(kitti_base_dir), "../../oxts/data/", os.path.splitext(all_files[i])[0]+".txt")
                                                                for i in range(len(all_files))])
    T_lid_imu = util_kitti._load_calib_lidar(os.path.join(os.path.dirname(kitti_base_dir), "../../calib/"))
    
    ground_truth_poses_lidar = np.zeros((len(ground_truth_poses), 4,4))
    pts = np.zeros((len(ground_truth_poses), 3))

    for i in range(len(ground_truth_poses)):
        ground_truth_poses_lidar[i] = T_lid_imu @ np.linalg.inv(ground_truth_poses[0]) @ ground_truth_poses[i]
        pts[i] = (ground_truth_poses_lidar[i] @ np.array([0,0,0,1]))[:3]
    
    plt.plot(pts[:,0], pts[:,1])
    plt.title("Kitti sequence")
    plt.show()
    print("Rigid transformation from LIDAR to IMU")
    print(T_lid_imu)

    return ground_truth_poses_lidar, pts

if __name__ == "__main__":
    # NEWER_COLLEGE_BASE_DIR = "/home/victor/Data/Stages/MIT/newer_college_dataset/ouster_scan/filenames.txt" # Newer college dataset
    
    KITTI_BASE_DIR = "/home/victor/Data/2011_09_26/2011_09_26_drive_0036_sync/velodyne_points/data/filenames.txt"
    all_files, folder =  load_all_filenames(KITTI_BASE_DIR) #NEWER_COLLEGE_BASE_DIR
    

    gt_poses, gt_pts = load_gt_imu_frame(KITTI_BASE_DIR, all_files)
    timestamps = kitti_read_timestamps(folder)

    prev_cloud, prev_cloud_npy = read_kitti(folder, all_files[0])
    prev_feats,_  = extract_fpfh(prev_cloud,VOXEL_SIZE)
    
    initial_transform = np.eye(4)
    origin = np.zeros(4)
    origin[-1] = 1

    positions = []
    positions.append(origin)

    initial_time = timestamps[0]
    start = datetime.now()

    idx = 0
    idx_max = 500#len(all_files)
    i = 1
    cur_pose = np.eye(4)

    while True:
        print("Now processing "+all_files[idx])
        teaser_solver = get_default_solver(NOISE_BOUND, estimate_scale = False)
        cur_cloud, cur_cloud_npy = read_kitti(folder, all_files[idx])
        print(np.max(np.max(cur_cloud_npy, 0) - np.min(cur_cloud_npy, 0)))
        # o3d.visualization.draw_geometries([cur_cloud])

        cur_feats, _ = extract_fpfh(cur_cloud,VOXEL_SIZE)

        A_corr, B_corr = find_correspondences(prev_feats, cur_feats)

        teaser_solver.solve(prev_cloud_npy[A_corr].T, cur_cloud_npy[B_corr].T)
        solution = teaser_solver.getSolution()
        transformation, scale = transform_from_solution(solution)
        icp_transf = icp(prev_cloud, cur_cloud, transformation)
            
        cur_pose = icp_transf @ cur_pose
        print(cur_pose)
        print(gt_poses[idx])
        print("---")
        # certify_solution(teaser_solver, solution.rotation, NOISE_BOUND)
        # visualization(prev_cloud, cur_cloud, icp_transf)

        positions.append(icp_transf @positions[-1])
        prev_cloud = cur_cloud
        prev_cloud_npy = cur_cloud_npy.copy()
        prev_feats = cur_feats.copy()
        
        delta_t = datetime.now() - start
        now_tmstamp = initial_time + delta_t
        idx, success = get_closest_cloud(now_tmstamp, idx, timestamps)

        if not success or idx > idx_max:
            break 
        
        i += 1

    positions = np.array(positions)
    plt.plot(positions[:,0], positions[:,1])
    # plt.plot(gt_pts[:,0], gt_pts[:,1])
    plt.show()
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(positions[:,:3] * NORMALIZATION_FACTOR)
