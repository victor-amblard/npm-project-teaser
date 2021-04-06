from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d 

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T, fpfh

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

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

def compute_fpfh_correspondences(cloud_a, cloud_b, voxel_size):
    print("Computing correspondences (FPFH)...")
    feats_a, _ = extract_fpfh(cloud_a, voxel_size)
    feats_b, _ =  extract_fpfh(cloud_b, voxel_size)
    corr_a, corr_b = find_correspondences(feats_a, feats_b)
    print("Found {} correspondences".format(feats_a.shape[0]))
    points_a = np.asarray(cloud_a.points).T
    points_b = np.asarray(cloud_b.points).T
    return points_a[:, corr_a], points_b[:, corr_b]
