"""
Homework 5
Submission Functions
"""
# import packages here
import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.ndimage import convolve2d
from scipy.signal import convolve2d
import helper as hlp
import cv2
"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
#8-point algorithm
def eight_point(pts1, pts2, M):
    T = np.diag([1/M, 1/M, 1])
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # Normalize the points using the transformation matrix T
    pts1_normalized = np.dot(pts1_homogeneous, T.T)
    pts2_normalized = np.dot(pts2_homogeneous, T.T)

    
    # Construct the A matrix
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1, _ = pts1_normalized[i]
        x2, y2, _ = pts2_normalized[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for the least squares solution of Af = 0
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint on F by zeroing out the smallest singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    
    # Unnormalize F
    F = np.dot(T.T, np.dot(F, T))
    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def similarity(target_window, candidate_window):
    return np.sqrt(np.sum((target_window - candidate_window) ** 2))



def epipolar_correspondences(im1, im2, F, pts1):
    # Convert points to homogeneous coordinates
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    
    # Compute the epipolar lines in the second image for each point in the first image
    epipolar_lines = np.dot(F, pts1_homogeneous.T)
    
    # Compute the normalized epipolar lines
    epipolar_lines_norm = epipolar_lines / np.sqrt(epipolar_lines[0]**2 + epipolar_lines[1]**2)
    
    # Initialize the list to store the corresponding points in the second image
    pts2 = []
    
    # Define the window size for similarity measurement
    window_size = 5
    
    for i in range(pts1.shape[0]):
        # Extract the coordinates of the point in the first image
        x, y = pts1[i]
        
        # Calculate the corresponding epipolar line in the second image
        a, b, c = epipolar_lines_norm[:, i]
        
        # Define the search range along the epipolar line
        y_min = max(0, int(round((-a * x - c) / b - window_size)))
        y_max = min(im2.shape[0], int(round((-a * x - c) / b + window_size + 1)))
        
        # Initialize variables to store the best match and its similarity score
        best_match = None
        best_score = float('-inf')
        
        # Search for the best match along the epipolar line
        for y0 in range(y_min, y_max):
            # Extract the window around the point in the second image
            window2 = im2[y0 - window_size:y0 + window_size + 1, x - window_size:x + window_size + 1]
            
            # Compute the similarity score between the windows
            window1 = im1[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1]
            score = np.sum(np.abs(window1 - window2))
            
            # Update the best match if necessary
            if score > best_score:
                best_match = (x, y0)
                best_score = score
        
        # Add the best match to the list of corresponding points
        pts2.append(best_match)
    
    # Convert the list of points to a numpy array
    pts2 = np.array(pts2)
    
    return pts2
"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    best_inliers = None
    best_E = None
    best_inlier_count = 0

    for i in range(1000):
        #Randomly sample minimal correspondences
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        E = np.dot(np.dot(K2.T, F), K1)

        #Compute the reprojection error and find inliers
        pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        errors = np.sum(pts2_homogeneous @ E @ pts1_homogeneous.T, axis=1)
        inliers = np.abs(errors) < 0.01
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inliers = inliers
            best_E = E
            best_inlier_count = inlier_count

    #Final essential matrix estimation using all inliers
    best_pts1 = pts1[best_inliers]
    best_pts2 = pts2[best_inliers]
    best_E, _ = cv2.findEssentialMat(best_pts1, best_pts2, K1, cv2.RANSAC, 0.999, 0.01)

    return best_E

def compute_camera_matrices(E, K1, K2):
    # Compute the first camera projection matrix P1
    I = np.eye(3)
    zero_translation = np.zeros((3, 1))
    extrinsic_matrix_P1 = np.hstack((I, zero_translation))
    P1 = K1 @ extrinsic_matrix_P1
    P2=hlp.camera2(E)
    return P1, P2


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""

def cost_function(X, P1, pts1):
    X = X.reshape(-1, 3)
    # Project 3D points into the first image
    P1_proj = P1 @ np.hstack((X, np.ones((X.shape[0], 1)))).T
    P1_proj /= P1_proj[2, :]
    P1_proj = P1_proj[:2, :].T
            
    # Compute the Euclidean error between projected points and given 2D points
    error = np.array((P1_proj - pts1))
    return np.linalg.norm(error)
def triangulate(P1, pts1, P2n, pts2):
    # Number of points
    num_points = pts1.shape[0]

    # Store the depth information for each 3D point
    depths = np.zeros((len(P2n), num_points))

    # Initialize variables to store the correct P2 and 3D points
    correct_P2 = None
    correct_points_3d = None
    max=0
    # Iterate over each candidate P2 matrix
    for i,P2 in enumerate(P2n):
        # Initialize the list to store the 3D points
        points_3d = []

        # Iterate over each point pair
        for j in range(num_points):
            # Get the corresponding points in homogeneous coordinates
            pt1 = np.hstack((pts1[j], 1))
            pt2 = np.hstack((pts2[j], 1))

            # Perform triangulation
            A = np.vstack((pt1[0] * P1[2] - P1[0], pt1[1] * P1[2] - P1[1], pt2[0] * P2[2] - P2[0], pt2[1] * P2[2] - P2[1]))
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1][:3] / Vt[-1][-1]

            # Append the 3D point
            points_3d.append(X)
            depth = np.sum(P1[2] * np.append(X, 1))
            if depth > 0:
                points_3d.append(X)

        # Convert the list of 3D points into a numpy array
        points_3d = np.array(points_3d)
        # Check if most points have positive depth
        if len(points_3d) > max:
            max = len(points_3d)
            correct_P2 = P2
            correct_points_3d = points_3d
        

    R2=t2=None
    # Check if a correct candidate P2 matrix was found
    if correct_P2 is not None and correct_points_3d is not None:
        # Extract the rotation (R2) and translation (t2) matrices from the correct P2 matrix
        R2 = correct_P2[:3, :3]
        t2 = correct_P2[:, 3]
        t2 = t2[:3] 

    # Save the camera extrinsic parameters (rotation and translation matrices) for both cameras (P1 and the correct P2) in an npz file
    np.savez("../data/extrinsics.npz", R1=P1[:, :3], t1=P1[:, 3], R2=R2, t2=t2)
    print("Reprojection error is",cost_function(correct_points_3d, P1, pts1, P2, pts2))   
    return correct_P2, correct_points_3d
    



def plot_correspondences(pts1, pts2):
    E=essential_matrix(F, K1, K2)
    pts1=np.load("../data/pts1-2.npy")
    pts2=epipolar_correspondences(img1, img2, f, pts1)
    print("Essential Matrix is", E)
    P1,P2=compute_camera_matrices(E, K1, K2)
    P1f,P2f=triangulate(P1, pts1, P2, pts2)
    plt.figure()
    plt.scatter(pts1[:, 0], pts1[:, 1], c='red', label='Image 1')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Point Correspondences in Image 1')
    plt.legend()

    # Plot the point correspondences in the second image (pts2)
    plt.figure()
    plt.scatter(pts2[:, 0], pts2[:, 1], c='blue', label='Image 2')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Point Correspondences in Image 2')
    plt.legend()

    # 3D scatter plot of the triangulated 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P2f[:, 0], P2f[:, 1], P2f[:, 2], c='green', marker='o', label='Triangulated 3D Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Triangulated 3D Points')
    ax.legend()

    plt.show()




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
  K1 = K1[:9]
  K1=K1.reshape(3,3)
  K2 = K2[:9]
  K2=K2.reshape(3,3)
  t1=t1[:3]
  t2=t2[:3]
  # Compute optical centers.
  c1 = -(K1 @ R1) @ np.linalg.inv(K1) @ t1
  c2 = -(K2 @ R2) @ np.linalg.inv(K2) @ t2

  # Compute new rotation matrices.
  r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
  r2 = np.cross(R1[:, 2], r1)
  r3 = np.cross(r1, r2)
  R01 = np.array([r1, r2, r3]).T
  R02 = R01

  # Compute new intrinsic parameters.
  K01 = K2
  K02 = K2

  # Compute new translation vectors.
  t1p = -R01 @ c1
  t2p = -R02 @ c2

  # Compute rectification matrices.
  M1 = K01 @ R01 @ np.linalg.inv(K1 @ R1)
  M2 = K02 @ R02 @ np.linalg.inv(K2 @ R2)

  return M1, M2, K01, K02, R01, R02, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def dist(im1, im2, d, win_size):
    w = (win_size - 1) // 2
    im2_shifted = np.roll(im2, -d, axis=1)  # Shift im2 by d to the right
    diff_squared = (im1 - im2_shifted) ** 2
    return convolve2d(diff_squared, np.ones((win_size, win_size)), mode='same')
def get_disparity(im1, im2, max_disp, win_size):
    disparity_map = np.zeros_like(im1)

    for y in range(im1.shape[0]):
        for x in range(im1.shape[1]):
            min_disp = max(0, x - max_disp)
            max_disp = min(im1.shape[1] - 1, x + max_disp)
            disparities = np.arange(min_disp, max_disp + 1)
            costs = dist(im1, im2, disparities, win_size)
            print(f"y={y}, x={x}, disparities={disparities}, costs={costs}")
            disparity_map[y, x] = disparities[np.argmin(costs[y, x])]

    return disparity_map



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # Compute baseline b and focal length f
    b = np.linalg.norm(t1 - t2)
    f = K1[0, 0]

    # Initialize depth map with zeros
    depthM = np.zeros_like(dispM, dtype=np.float32)

    # Calculate depth map using the provided formula
    mask = dispM != 0
    depthM[mask] = b * f / dispM[mask]

    return depthM

#Localisation code


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # Ensure points are in homogeneous coordinates
    X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))
    x_homogeneous = np.hstack((x, np.ones((x.shape[0], 1))))

    # Perform Direct Linear Transform (DLT) algorithm
    A = []
    for i in range(len(X)):
        X_row = X_homogeneous[i]
        x_row = x_homogeneous[i]
        A.append([-X_row[0], -X_row[1], -X_row[2], -1, 0, 0, 0, 0, x_row[0] * X_row[0], x_row[0] * X_row[1], x_row[0] * X_row[2], x_row[0]])
        A.append([0, 0, 0, 0, -X_row[0], -X_row[1], -X_row[2], -1, x_row[1] * X_row[0], x_row[1] * X_row[1], x_row[1] * X_row[2], x_row[1]])

    A_matrix = np.array(A)
    _, _, Vh = np.linalg.svd(A_matrix)
    P = Vh[-1].reshape(3, 4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    K = P[:, :3]
    K, E = np.linalg.qr(K)
    R = np.linalg.inv(K) @ P[:, :3]
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t




