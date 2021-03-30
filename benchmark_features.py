import open3d as o3d 
import time
import sys
sys.path.append('FCGF')
import numpy as np 
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import math
import MinkowskiEngine as ME
# FCGF import
from FCGF.util.pointcloud import make_open3d_point_cloud, make_open3d_feature_from_numpy
from FCGF.util.misc import extract_features
from FCGF.model.resunet import ResUNetBN2C
# TEASER++ import 
import teaserpp_python
from open3d.open3d.geometry import voxel_down_sample, estimate_normals
import torch

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

def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        import pdb; pdb.set_trace()
        return 99999

def print_error(est_mat, gt_mat):
    """
    Compute difference between two 4-by-4 SE3 transformation matrix
    """
    R_gt = gt_mat[:3,:3]
    R_est = est_mat[:3,:3]
    rot_error = get_angular_error(R_gt, R_est)

    t_gt = gt_mat[:,-1]
    t_est = est_mat[:,-1]
    trans_error = np.linalg.norm(t_gt - t_est)


    print("ERROR: {:.2f} degrees {:.3f}m".format(rot_error, trans_error))

    return rot_error, trans_error

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 

# Code snippet taken from open3d documentation
def ransac_registration(source_down, target_down, source_feats, target_feats, voxel_size):

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_feats, target_feats, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))

    return result.transformation

def teaser_registration(correspondences_source, correspondences_target, voxel_size):
    '''
        Returns a SE(3) matrix and a certificate
    '''

    noise_bound = voxel_size
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False

    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 1000
    solver_params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(correspondences_source, correspondences_target)
    solution = solver.getSolution()
    '''
    tims_a = solver.getMaxCliqueSrcTIMs()
    for i in range(tims_a.shape[1]):
        tims_a[:,i ]= tims_a[:,i] / np.linalg.norm(tims_a[:,i])
    
    tims_b = solver.getMaxCliqueDstTIMs()
    for i in range(tims_b.shape[1]):
        tims_b[:,i]= tims_b[:,i] / np.linalg.norm(tims_b[:,i])

    theta = solver.getRotationInliersMask().astype(int)
    theta[np.where(theta == 0)] = -1

    certifier_params = teaserpp_python.DRSCertifier.Params()
    certifier_params.cbar2 = 1.0
    certifier_params.noise_bound = 2*noise_bound 
    certifier_params.sub_optimality = 1e-3
    certifier_params.max_iterations = 1e2
    certifier_params.gamma_tau = 1.8
    
    certifier = teaserpp_python.DRSCertifier(certifier_params)

    certificate = certifier.certify(solution.rotation, tims_a, tims_b, theta)

    return Rt2T(solution.rotation, solution.translation), certificate.is_optimal
    '''
    return Rt2T(solution.rotation, solution.translation), None

# FPFH features
def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
  radius_feature = voxel_size * 5
  fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

  return np.array(fpfh.data).T, fpfh

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

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]
  

  return corres_idx0, corres_idx1


if __name__ == "__main__":

    fn_a = "./TEASER-plusplus/examples/example_data/3dmatch_sample/cloud_bin_36.ply"
    fn_b = "./TEASER-plusplus/examples/example_data/3dmatch_sample/cloud_bin_2.ply"

    gt_mat = load_gt_transformation(2,36,"./TEASER-plusplus/examples/example_data/3dmatch_sample/gt.log")

    pcd_a = o3d.io.read_point_cloud(fn_a)
    pcd_b = o3d.io.read_point_cloud(fn_b)

    voxel_size = 0.06

    pcd_a_ds = voxel_down_sample(pcd_a, voxel_size)
    pcd_a_npy = np.asarray(pcd_a_ds.points)
    pcd_b_ds = voxel_down_sample(pcd_b, voxel_size)
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

    n_mc = 1
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