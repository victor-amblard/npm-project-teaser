import os 
import open3d as o3d 
import numpy as np
import copy
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

def generate_random_transformation():
    '''
        Generate random SE(3)
    '''
    rot = R.random(random_state=42).as_matrix()
    tr = np.random.normal(0,1,3)

    transf = np.zeros((4,4))
    transf[:3, :3] = rot
    transf[:3, 3] = tr.T 
    transf[3,3] = 1

    return transf 

def generate_outliers(N): 
    dirs = np.random.normal(0,1,3*N).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=-1)[:,None]
    dirs = dirs*np.random.normal(0,1,N)[:, None]
    return dirs 

def generate_noise(sigma, n_points):
    random_noise = np.random.normal(0, sigma, n_points * 3).reshape(n_points, 3)

    return random_noise

def incomplete_scan(scan, n_points, p=0.95):
    valid_idx = np.random.choice(range(n_points), int(p*n_points))

    return scan[valid_idx]

if __name__ == "__main__":
    '''
    
    Usage
    python generate_noise.py data/model_bunny.ply data/output.ply --scale 1 --noise 0.03 --outliers 0.4
    
    '''
    parser = ArgumentParser()

    parser.add_argument('input', default = '/home/victor/Documents/model_bunny.ply', help='Input file')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--scale', help='Add a random scale')
    parser.add_argument('--noise', help='Add some noise')
    parser.add_argument('--partial_overlap', help='Add some noise')
    parser.add_argument('--outliers', help='Add outliers')

    args = parser.parse_args()
    
    input_cloud = args.input
    output_cloud = args.output 

    i_cloud = o3d.io.read_point_cloud(input_cloud)
    o_cloud = copy.deepcopy(i_cloud)

    transf = generate_random_transformation()

    o_cloud.transform(transf) # Rotate + translate
    points = np.asarray(o_cloud.points)

    if args.scale:
        # scale = max(0.3, 3*np.random.random())
        points *= float(args.scale)
        transf[3,3] = float(args.scale)
    if args.partial_overlap:
        points = incomplete_scan(points, points.shape[0], float(args.partial_overlap))

    if args.noise:
        noise = generate_noise(float(args.noise), points.shape[0])
        points += noise

    if args.outliers:
        centroid = np.mean(points, axis=0)
        outliers = generate_outliers(int(float(args.outliers)*points.shape[0])) + centroid[None, :]
        points = np.vstack((points, outliers))

    o_cloud.points = o3d.utility.Vector3dVector(points)
    
    i_cloud.paint_uniform_color([0,0,1])
    o_cloud.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([i_cloud, o_cloud])
    
    o3d.io.write_point_cloud(args.output, o_cloud)
    np.savetxt(os.path.join(os.path.dirname(output_cloud), "ground_truth_transform.txt"), transf)
