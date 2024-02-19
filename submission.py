"""
Homework 5
Submission Functions
"""
import numpy as np
import cv2
import helper as hlp
import scipy
# import packages here

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""

def eight_point(pts1, pts2, M):
    T = np.diag([1 / M, 1 / M, 1])
    
    normpts1 = pts1/M
    normpts2 = pts2/M
    
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x, y = normpts1[i]
        xp, yp = normpts2[i]
        A[i] = [x*xp, x*yp, x, y*xp, y*yp, y, xp, yp, 1]
    
    _, _, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)
    
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = u @ np.diag(s) @ vh
    
    F = hlp.refineF(F, normpts1, normpts2)
    
    F_denormal = T.T @ F @ T
    
    return F_denormal



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1):
    window_size = 10
    pts2 = []  # To store the corresponding points in the second image
    for pt1 in pts1:
        # Homogenize the point
        pt1_hom = np.array([pt1[0], pt1[1], 1])
        # Compute the epipolar line in the second image
        l = F @ pt1_hom
        
        # Find two points on the epipolar line by intersecting it with the image borders
        # These points are used to sample points along the line within the image bounds
        x0, x1 = 0, im2.shape[1] - 1
        y0 = -(l[2] + l[0] * x0) / l[1] if l[1] != 0 else -1
        y1 = -(l[2] + l[0] * x1) / l[1] if l[1] != 0 else -1
        
        # Ensure y0 and y1 are within image bounds
        y0 = np.clip(y0, 0, im2.shape[0] - 1)
        y1 = np.clip(y1, 0, im2.shape[0] - 1)
        
        # Generate candidate points along the line segment between (x0, y0) and (x1, y1)
        num_points = 100  # Number of points to sample along the epipolar line
        x_coords = np.linspace(x0, x1, num_points)
        y_coords = np.linspace(y0, y1, num_points)
        
        best_score = np.inf
        best_pt = None
        for x, y in zip(x_coords, y_coords):
            x = int(round(x))
            y = int(round(y))
            # Compute the similarity score
            score = compare_windows(im1, im2, pt1, (x, y), window_size)
            if score < best_score:
                best_score = score
                best_pt = (x, y)
        if best_pt is not None:
            pts2.append(best_pt)
    
    return np.array(pts2)


def compare_windows(im1, im2, pt1, pt2, window_size):
    half_window = window_size // 2
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    # Define window bounds, taking care not to exceed image dimensions
    x1_min, x1_max = max(pt1[0]-half_window, 0), min(pt1[0]+half_window+1, w1)
    y1_min, y1_max = max(pt1[1]-half_window, 0), min(pt1[1]+half_window+1, h1)
    x2_min, x2_max = max(pt2[0]-half_window, 0), min(pt2[0]+half_window+1, w2)
    y2_min, y2_max = max(pt2[1]-half_window, 0), min(pt2[1]+half_window+1, h2)

    # Extract windows
    window1 = im1[y1_min:y1_max, x1_min:x1_max]
    window2 = im2[y2_min:y2_max, x2_min:x2_max]

    # Pad the windows if they're near the edge of the image
    pad_width1 = ((max(0, half_window - pt1[1]), max(0, pt1[1] + half_window + 1 - h1)),
                  (max(0, half_window - pt1[0]), max(0, pt1[0] + half_window + 1 - w1)),
                  (0, 0))
    pad_width2 = ((max(0, half_window - pt2[1]), max(0, pt2[1] + half_window + 1 - h2)),
                  (max(0, half_window - pt2[0]), max(0, pt2[0] + half_window + 1 - w2)),
                  (0, 0))

    window1_padded = np.pad(window1, pad_width1, mode='constant', constant_values=0)
    window2_padded = np.pad(window2, pad_width2, mode='constant', constant_values=0)

    # Compute the normalized correlation coefficient
    product = np.mean((window1_padded - np.mean(window1_padded)) * (window2_padded - np.mean(window2_padded)))
    stds = np.std(window1_padded) * np.std(window2_padded)
    if stds == 0:
        return float('inf')  # Avoid division by zero; consider this a bad match
    score = -product / stds  # Negative because lower scores should indicate better matches
    
    return score



"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""

def triangulate(P1, pts1, P2, pts2):
    """
    Triangulate the 3D point positions from corresponding 2D points in two images and compute the re-projection error.

    Parameters:
    P1 -- 3x4 projection matrix for the first camera
    pts1 -- N x 2 matrix of 2D points in the first image
    P2 -- 3x4 projection matrix for the second camera
    pts2 -- N x 2 matrix of 2D points in the second image

    Returns:
    pts3d -- N x 3 matrix of 3D points
    reprojection_error -- Mean Euclidean error of the re-projected points
    """
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))
    errors = []

    for i in range(num_points):
        # Construct matrix A from the corresponding points and projection matrices.
        A = np.array([
            pts1[i, 0] * P1[2, :] - P1[0, :],
            pts1[i, 1] * P1[2, :] - P1[1, :],
            pts2[i, 0] * P2[2, :] - P2[0, :],
            pts2[i, 1] * P2[2, :] - P2[1, :]
        ])

        # Solve for X using SVD, which minimizes the error.
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[-1]  # Dehomogenize

        pts3d[i, :] = X[:3]

        # Project the 3D point back to both images.
        X_homogeneous = np.append(pts3d[i, :], 1)  # Make homogeneous
        projected_pt1 = (P1 @ X_homogeneous)[:2] / (P1 @ X_homogeneous)[2]  # Dehomogenize
        projected_pt2 = (P2 @ X_homogeneous)[:2] / (P2 @ X_homogeneous)[2]  # Dehomogenize

        # Compute Euclidean error for this point for both cameras.
        error1 = np.linalg.norm(projected_pt1 - pts1[i, :])
        error2 = np.linalg.norm(projected_pt2 - pts2[i, :])
        
        # Accumulate errors from both views.
        errors.append(error1)
        errors.append(error2)

    # Compute mean reprojection error.
    reprojection_error = np.mean(errors)

    return pts3d, reprojection_error

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""

def rectify_pair(K1, K2, R1, R2, t1, t2):
    # compute camera centers
    c1 = -1 * np.linalg.inv(K1 @ R1) @ K1 @ t1
    c2 = -1 * np.linalg.inv(K2 @ R2) @ K2 @ t2

    r1 = c1 - c2/ np.linalg.norm(c1 - c2)
    r2 = np.cross(r1.flatten(), R1[2, :])
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r2, r1.flatten()).reshape(3, 1)
    r2 = r2.reshape(3, 1)

    R = np.hstack((r1, r2, r3)).T 
    R1_rect, R2_rect = R, R
    R2_rect = R

    K1_rect, K2_rect = K2, K2

    t1_rect = -1 * R @ c1
    t2_rect = -1 * R @ c2

    M1 = (K1_rect @ R1_rect) @ np.linalg.inv(K1 @ R1)
    M2 = (K2_rect @ R2_rect) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1_rect, K2_rect, R1_rect, R2_rect, t1_rect, t2_rect


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    dispM = np.zeros_like(im1)
    filter = np.ones((win_size, win_size))
    dist = np.zeros((im1.shape[0],im1.shape[1],max_disp+1))
    for d in range(max_disp + 1):
        rolled = np.roll(im2, d, axis=1)
        difference_squared = np.square(rolled - im1)
        dist[:,:,d] = scipy.signal.convolve2d(difference_squared, filter, mode="same")
    dispM = np.argmin(dist, axis=2)
    return dispM

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -1 * np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -1 * np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]
    notzero = dispM != 0

    depthM = np.zeros_like(dispM, dtype=float)
    depthM[notzero] = (b * f) / dispM[notzero]
    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
