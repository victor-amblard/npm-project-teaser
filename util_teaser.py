import numpy as np
import math 
import teaserpp_python


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


def transform_from_solution(solution):
    T = np.identity(4)
    T[:3,:3] = solution.rotation
    T[:3,3] = solution.translation
    
    return T, solution.scale

def get_default_solver(noise_bound, estimate_scale = False):
    
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = estimate_scale
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    return solver

def certify_solution(solver, R, noise_bound):
    '''
        Returns a boolean (corresponding to the global optimality certification)
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

    certificate = certifier.certify(R, tims_a, tims_b, theta)
    print("Certified solution: "+str(certificate.is_optimal))
    
    return certificate.is_optimal 