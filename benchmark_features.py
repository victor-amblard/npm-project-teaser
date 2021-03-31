import open3d as o3d 
import time
import sys
sys.path.append('FCGF')
import numpy as np 
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import math
import MinkowskiEngine as ME
from open3d.open3d.geometry import voxel_down_sample, estimate_normals
import torch

from util.util_features import find_correspondences, extract_fpfh
from util.util_teaser import get_default_solver, transform_from_solution, certify_solution, print_error, visualize, NormalizeCloud
from util.util_ransac import ransac_registration
# TEASER++ import 
import teaserpp_python
# FCGF import
from FCGF.util.pointcloud import make_open3d_point_cloud, make_open3d_feature_from_numpy
from FCGF.util.misc import extract_features
from FCGF.model.resunet import ResUNetBN2C


def icp(cloud_a, cloud_b, transf):
    icp_sol = o3d.registration.registration_icp(
            cloud_a, cloud_b, 0.05, transf,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))

    return icp_sol.transformation

# From TEASER++ repo
def load_gt_transformation(fragment1_idx, fragment2_idx, gt_log_path):
    """
    Load gt transformation
    """
    with open(gt_log_path) as f:
        content = f.readlines()

    for i in range(len(content)):
        tokens = [k.strip() for k in content[i].split(" ")]
        if tokens[0] == str(fragment1_idx) and tokens[1] == str(fragment2_idx):

            def line_to_list(line):
                return [float(k.strip()) for k in line.split("\t")[:4]]

            gt_mat = np.array([line_to_list(content[i+1]),
                               line_to_list(content[i+2]), 
                               line_to_list(content[i+3]), 
                               line_to_list(content[i+4])])
            return gt_mat
    return None


# FCGF features
def extract_fcgf(pcd, voxel_size, model):
    xyz_down, feature = extract_features(
        model,
        xyz=np.array(pcd.points),
        voxel_size=voxel_size,
        device='cpu',
        skip_check=True)

    return xyz_down, make_open3d_feature_from_numpy(feature.detach().cpu().numpy() )

def load_fcgf_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to('cpu')

    return model


def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 
    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)
    print(test_NN_idx)
    print(ref_NN_idx)
    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    print(ref_match_idx)

    return ref_match_idx, ref_NN_idx[ref_match_idx]

def teaser_registration(corr_a, corr_b, voxel_size): 
    solver = get_default_solver(voxel_size)
    solver.solve(corr_a, corr_b)
    solution = solver.getSolution()
    T, _ = transform_from_solution(solution)
    # cert = certify_solution(solver, solution.rotation, voxel_size)

    return T,None# cert 

def test_reconstruction(all_filenames, voxel_size): 
    start = time.time()
    fn_start = all_filenames[0]
    pcd_a = o3d.io.read_point_cloud(fn_start)
    
    for fn in all_filenames[1:]:
        pcd_a_ds = voxel_down_sample(pcd_a, voxel_size)
        pcd_a_npy = np.asarray(pcd_a_ds.points)
        pcd_a_ds.paint_uniform_color([0,0,1])
        fpfh_feats_a, fpfh_raw_a = extract_fpfh(pcd_a_ds, voxel_size)

        pcd_b = o3d.io.read_point_cloud(fn)
        pcd_b_ds = voxel_down_sample(pcd_b, voxel_size)
        pcd_b_npy = np.asarray(pcd_b_ds.points)
        pcd_b_ds.paint_uniform_color([0,1,0])
        fpfh_feats_b, fpfh_raw_b = extract_fpfh(pcd_b_ds, voxel_size)
        a_corr, b_corr = find_correspondences(fpfh_feats_a, fpfh_feats_b)

        T_teaser, certificate = teaser_registration(pcd_a_npy[a_corr].T, pcd_b_npy[b_corr].T,voxel_size)
        tr = icp(pcd_a_ds, pcd_b_ds, T_teaser)
        # visualize(pcd_a, pcd_b, np.linalg.inv(tr))

        pcd_a += pcd_b.transform(np.linalg.inv(tr))

        pcd_a = voxel_down_sample(pcd_a, 0.01)
        print(pcd_a.points)
    o3d.visualization.draw_geometries([pcd_a])
    end = time.time()
    print(end-start)
    o3d.io.write_point_cloud('./data/kitchen_reconstruction.ply', pcd_a)
    
if __name__ == "__main__":
    voxel_size = 0.1


    # Surface reconstruction POC
    test_reconstruction(["/home/victor/Data/3DMatch/kitchen/cloud_bin_{}.ply".format(i) for i in range(15)], voxel_size)

    fn_a = "/home/victor/Data/3DMatch/kitchen/cloud_bin_0.ply"
    fn_b = "/home/victor/Data/3DMatch/kitchen/cloud_bin_2.ply"

    gt_mat = load_gt_transformation(0,2,"/home/victor/Data/3DMatch/kitchen/gt.log")

    pcd_a = o3d.io.read_point_cloud(fn_a)
    pcd_b = o3d.io.read_point_cloud(fn_b)


    pcd_a_ds = voxel_down_sample(pcd_a, voxel_size)
    pcd_b_ds = voxel_down_sample(pcd_b, voxel_size)

    pcd_a_npy = np.asarray(pcd_a_ds.points)
    pcd_b_npy = np.asarray(pcd_b_ds.points)

    pcd_a_ds.paint_uniform_color([0,0,1])
    pcd_b_ds.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pcd_a_ds, pcd_b_ds])

    print("FPFH features")
    fpfh_feats_a, fpfh_raw_a = extract_fpfh(pcd_a_ds, voxel_size)
    fpfh_feats_b, fpfh_raw_b = extract_fpfh(pcd_b_ds, voxel_size)
    a_corr, b_corr = find_correspondences(fpfh_feats_a, fpfh_feats_b)
    
    T_teaser, certificate = teaser_registration(pcd_a_npy[a_corr].T, pcd_b_npy[b_corr].T,voxel_size)
    print_error(T_teaser, gt_mat)
    
    visualize(pcd_a, pcd_b, np.linalg.inv(T_teaser))


    n_mc = 5
    mean_rot_err, mean_trans_err, mean_runtime = 0,0,0 
    for mc in range(n_mc):
        start = time.time()
        T_ransac = ransac_registration(pcd_a_ds, pcd_b_ds, fpfh_raw_a, fpfh_raw_b, voxel_size)
        end = time.time()
        rot_err, trans_err = print_error(T_ransac, gt_mat)
        mean_rot_err += rot_err
        mean_trans_err += trans_err
        mean_runtime += (end - start)
    
    print("Average over 10 runs (RANSAC): {:.1f} deg, {:.2f}m ({:.2f}s)".format(mean_rot_err/n_mc, mean_trans_err/n_mc, mean_runtime/n_mc))
    print("FCGF features")

    model_fcgh = load_fcgf_model("./FCGF/ResUNetBN2C-16feat-3conv.pth")
    xyz_a, fcgh_feats_a = extract_fcgf(pcd_a, voxel_size, model_fcgh)
    xyz_b, fcgh_feats_b = extract_fcgf(pcd_b, voxel_size, model_fcgh)
    # print("n1 = {}, n2 = {}".format(fcgh_feats_a.shape[0], fcgh_feats_b.shape[0]))
    new_pcloud_a, new_pcloud_b = make_open3d_point_cloud(xyz_a), make_open3d_point_cloud(xyz_b)

    a_corr, b_corr = find_mutually_nn_keypoints(new_pcloud_a, new_pcloud_b,fcgh_feats_a, fcgh_feats_b)

    print(a_corr.shape)
    T_teaser, certificate = teaser_registration(xyz_a[a_corr].T, xyz_b[b_corr].T,voxel_size)
    print(T_teaser)
    print_error(T_teaser, gt_mat)
    T_ransac = ransac_registration(new_pcloud_a, new_pcloud_b, fcgh_feats_a, fcgh_feats_b, voxel_size)
    print(T_ransac)
    print_error(T_ransac, gt_mat)
    print("3D Smooth net features")


    # ransac_registration()