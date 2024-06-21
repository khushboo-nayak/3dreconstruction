#importing libraries
import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import scipy.optimize
from scipy import ndimage
import cv2
from mpl_toolkits.mplot3d import Axes3D
import submission as sub
def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]

    return e1, e2
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


def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[1] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()

def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))

    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))
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

def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, sd = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[0] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        pc = np.array([[xc, yc]])
        p2 = epipolar_correspondences(I1, I2, F, pc)
        ax2.plot(p2[0,0], p2[0,1], 'ro', markersize=8, linewidth=2)
        plt.draw()

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
        

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)

    return M2s



def compute_camera_matrices(E, K1, K2):
    # Compute the first camera projection matrix P1
    I = np.eye(3)
    zero_translation = np.zeros((3, 1))
    extrinsic_matrix_P1 = np.hstack((I, zero_translation))
    P1 = K1 @ extrinsic_matrix_P1
    P2=camera2(E)
    return P1, P2

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
          R2 = correct_P2[:3, :3] - np.eye(3)
          t2 = correct_P2[-1, :]
    R1 = P1[:3, :3] - np.eye(3)
    t1 = P1[-1, :]
        

    # Save the camera extrinsic parameters (rotation and translation matrices) for both cameras (P1 and the correct P2) in an npz file
    np.savez("../data/extrinsics.npz", R1=R1, t1=t1, R2=R2, t2=t2)
    print("Reprojection error is",cost_function(correct_points_3d, P1, pts1))   
    return correct_P2, correct_points_3d
    
        


        
if __name__ == "__main__":
    #Loading data
    pts1=np.load("../data/pts1.npy")
    pts2=np.load("../data/pts2.npy")
    K1= np.load("../data/K1.npy")
    K2= np.load("../data/K2.npy")
    img1 = plt.imread("../data/im1.png")
    img2 = plt.imread("../data/im2.png")
    height, width = img1.shape[:2]
    M= max(width, height)
    F=eight_point(pts1, pts2, M)
    f=refineF(F, pts1, pts2)
    E=essential_matrix(F, K1, K2)
    pts1=np.load("../data/pts1-2.npy")
    pts2=epipolar_correspondences(img1, img2, f, pts1)
    epipolarMatchGUI(img1, img2, f)
    print("Essential Matrix is", E)
    P1,P2=compute_camera_matrices(E, K1, K2)
    P1f,P2f=triangulate(P1, pts1, P2, pts2)
    # Plot the point correspondences in the first image (pts1)
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
    
 

