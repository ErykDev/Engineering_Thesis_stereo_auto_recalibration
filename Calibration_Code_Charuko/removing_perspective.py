import cv2 
import numpy as np

import cv2.aruco as aruco

import time

dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

board_shape = [8, 11]
img_shape = (3072, 4096)

board = aruco.CharucoBoard(board_shape, 0.044, 0.034, dictionary)

frame = cv2.imread('/home/eryk-dev/Downloads/master/1690359208.2684517.jpg')

K1 = np.loadtxt('Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
D1 = np.loadtxt('dist_1.txt', dtype=float)



#roi can be ignored sine we using alpha=0
K1_opt, roi_1 = cv2.getOptimalNewCameraMatrix(K1, D1, img_shape, 0, img_shape)

start = time.time_ns()

frame = cv2.undistort(frame, K1, D1, None, K1_opt)

undistort_time = time.time_ns() - start


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

corners, ids, rejectedCorners = arucoDetector.detectMarkers(gray)

ret, corners, ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

def calculateImgPointsObjPoints(ids, corners, board):
    imgPoints, objPoints = [], []

    for ids, corner in zip(ids, corners):
        op, ip = board.matchImagePoints(
            np.array(corner, np.float32),
            np.array(ids, np.int32))
        imgPoints.append(ip)
        objPoints.append(op)

    return imgPoints, objPoints


imgPoints, objPoints = calculateImgPointsObjPoints(ids, corners, board)

imgPoints = np.squeeze(np.array(imgPoints))
objPoints = np.squeeze(np.array(objPoints)) * 1000.0 * 2

# align left bord corner to center of the image
objPoints = objPoints + (frame.shape[1] / 2, frame.shape[0] / 2, 0)

# align center of the board to center of the image 
objPoints = objPoints - (0.044 * 3 * 1000.0  * 2, 0.044 * 6 * 1000.0, 0  * 2)

# Compute the perspective transformation matrix
perspective_matrix, _ = cv2.findHomography(imgPoints, objPoints)

start = time.time_ns()


# Apply the perspective transformation to remove perspective distortion
corrected_image = cv2.warpPerspective(frame, perspective_matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_CONSTANT)

end = time.time_ns() - start

print(undistort_time)
print(end)


print(end +  undistort_time)

corrected_image = cv2.resize(corrected_image, (int(frame.shape[1] / 4.5), int(frame.shape[0]/ 4.5)))

# Display the undistorted and perspective-corrected image
cv2.imshow('Undistorted and Perspective Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()