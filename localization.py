import numpy as np
import cv2
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



if __name__ == "__main__":
    aruco_tag_size = 10 
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    while True:
        if ids is not None:
            # If ArUco tag is detected, draw the markers and estimate the camera pose
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            x=[]
            X=[]
            for i in range(len(ids)):
                x.append(corners[i][0])
                X.append(tvecs[i][0])
            camera_matrix = estimate_pose(x, X)


            # Calculate the distance of the camera from the ArUco tag using the tag's known size
            focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
            distance = (aruco_tag_size * focal_length) / (corners[0][0][1][0] - corners[0][0][0][0])

            # Estimate camera pose (rotation and translation vectors)
            K,rvecs, tvecs = estimate_params(camera_matrix)

            # Convert rotation vector to Euler angles (roll, pitch, yaw)
            rvec = rvecs[0][0]
            rmat, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvecs[0].reshape(3, 1))))[6]

            # Print camera height, distance to tag, and orientation (roll, pitch, yaw)
            print("Camera Height: {:.2f} cm".format(tvecs[0][0][1]))
            print("Distance to ArUco Tag: {:.2f} cm".format(distance))
            print("Orientation (Roll, Pitch, Yaw): {:.2f}, {:.2f}, {:.2f}".format(np.degrees(roll), np.degrees(pitch), np.degrees(yaw)))

        # Show the frame with ArUco tag detection
        cv2.imshow('ArUco Tag Detection', frame)
    
        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
