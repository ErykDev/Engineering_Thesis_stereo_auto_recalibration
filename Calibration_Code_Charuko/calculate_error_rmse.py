import cv2
import numpy as np
import argparse
import os
import os.path

from calibration_utils import *
from cv2 import aruco

class CameraCoeff :
    def __init__(self, K1, K2, D1, D2, R, T, Type):
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1
        self.D2 = D2
        self.R = R
        self.T = T
        self.Type = Type
    

def load_camera_coeff(root_path, Type):
    K1 = np.loadtxt(root_path + '/Intrinsic_mtx_1.txt', dtype=float).reshape((3, 3))
    K2 = np.loadtxt(root_path + '/Intrinsic_mtx_2.txt', dtype=float).reshape((3, 3))

    R = np.loadtxt(root_path + '/R.txt', dtype=float)
    T = np.loadtxt(root_path + '/T.txt', dtype=float)

    D1 = np.loadtxt(root_path + '/dist_1.txt', dtype=float)
    D2 = np.loadtxt(root_path + '/dist_2.txt', dtype=float)

    return CameraCoeff(K1, K2, D1, D2, R, T, Type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='error_collector')
    parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                        help='Array of cameras_ids')

    parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1024, 768),
                        help='Expected frame shape w h')

    parser.add_argument('--board_shape', nargs='+', type=int, default=(7, 5),
                        help='expected board shape')

    parser.add_argument('--checker_size_mm', type=int, default=38,
                        help='expected board shape')

    parser.add_argument('--marker_size_mm', type=int, default=28,
                        help='expected board shape')

    args = parser.parse_args()


    scene_path = '/home/eryk-dev/Desktop/Engineering_Thesis_stereo_auto_recalibration/Collected_Scenes/scena2/'


    starting_calibration_path = scene_path + 'Starting_Calibration/'
    starting_calibration_path = starting_calibration_path + os.listdir(starting_calibration_path)[0] + '/'


    ending_calibration_path = scene_path + 'ENDING_Calibration/'
    ending_calibration_path = ending_calibration_path + os.listdir(ending_calibration_path)[0] + '/'


    collected_frames_path = scene_path + 'Frames/'



    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters()
    arucoDetector = aruco.ArucoDetector(dictionary, parameters)


    checker_size_m = args.checker_size_mm * 0.001
    marker_size_m = args.marker_size_mm * 0.001

    corner_ids = cornerIds(args.board_shape)


    board = aruco.CharucoBoard(args.board_shape, checker_size_m, marker_size_m, dictionary)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    img_shape = args.cameras_frame_shape

    collected_frames = 0

    decimator = 0

    allCornersLeft = []
    allCornersRight = []
    allIdsLeft = []
    allIdsRight = []



    frames_left = os.listdir(ending_calibration_path + '/cam2/')
    frames_left.sort(key=lambda x: int(x.split('.')[0]))

    frames_right = os.listdir(ending_calibration_path + '/cam4/')
    frames_right.sort(key=lambda x: int(x.split('.')[0]))


    #Main loop
    for left_path, right_path in zip(frames_left, frames_right):   
        frame_left = cv2.imread(ending_calibration_path + 'cam2/'+ left_path)
        frame_right = cv2.imread(ending_calibration_path + 'cam4/'+ right_path)
        
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        corners_left,  ids_left,  rejectedCorners1 = arucoDetector.detectMarkers(gray_left)
        corners_right, ids_right, rejectedCorners2 = arucoDetector.detectMarkers(gray_right)
        
        aruco.refineDetectedMarkers(gray_left, board, corners_left, ids_left, rejectedCorners1)
        aruco.refineDetectedMarkers(gray_right, board, corners_right, ids_right, rejectedCorners2)
        
        res2_l  = aruco.interpolateCornersCharuco(corners_left, ids_left, gray_left, board)
        res2_r = aruco.interpolateCornersCharuco(corners_right, ids_right, gray_right, board)

        if res2_l[1] is not None and res2_r[1] is not None and res2_l[2] is not None and len(res2_l[1])>3:
            allCornersLeft.append(res2_l[1])
            allCornersRight.append(res2_r[1])

            allIdsLeft.append(res2_l[2])
            allIdsRight.append(res2_r[2])

            collected_frames += 1 

    shared_corners_l, shared_ids_l, shared_corners_r, \
    shared_ids_r = getSharedFetures(
        allCornersLeft, 
        allIdsLeft, 
        allCornersRight, 
        allIdsRight, 
        board)

    imgPoints_l, objPoints_l = calculateImgPointsObjPoints(shared_ids_l, shared_corners_l, board)
    imgPoints_r, objPoints_r = calculateImgPointsObjPoints(shared_ids_r, shared_corners_r, board)

    stereo_module_coeff = load_camera_coeff('/home/eryk-dev/Desktop/Engineering_Thesis_stereo_auto_recalibration/Collected_Scenes/scena2/Recalibration_Values/', 'pinhole')

rvec_l = []
tvec_l = []

rvec_r = []
tvec_r = []

# SolvePnP for left camera
for op, ip in zip(objPoints_l, imgPoints_l):
    ret, rvec, tvec = cv2.solvePnP(op, ip, stereo_module_coeff.K1, stereo_module_coeff.D1)
    rvec_l.append(rvec)
    tvec_l.append(tvec)

# SolvePnP for right camera
for op, ip in zip(objPoints_r, imgPoints_r):
    ret, rvec, tvec = cv2.solvePnP(op, ip, stereo_module_coeff.K2, stereo_module_coeff.D2)
    rvec_r.append(rvec)
    tvec_r.append(tvec)

# Calculate total reprojection error
total_error = 0
total_points = 0

def calc_rms_stereo(objectpoints, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T):
    tot_error = 0
    total_points = 0

    for i, objpoints in enumerate(objectpoints):
        # calculate world <-> cam1 transformation
        _, rvec_l, tvec_l,_ = cv2.solvePnPRansac(objpoints, imgpoints_l[i], A1, D1)

        # compute reprojection error for cam1
        rp_l, _ = cv2.projectPoints(objpoints, rvec_l, tvec_l, A1, D1)
        tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
        total_points += len(objpoints)

        # calculate world <-> cam2 transformation
        rvec_r, tvec_r  = cv2.composeRT(rvec_l,tvec_l,cv2.Rodrigues(R)[0],T)[:2]

        # compute reprojection error for cam2
        rp_r,_ = cv2.projectPoints(objpoints, rvec_r, tvec_r, A2, D2)
        tot_error += np.square(imgpoints_r[i] - rp_r).sum()
        total_points += len(objpoints)

    mean_error = np.sqrt(tot_error/total_points)
    return mean_error

print(f"Stereo re-projection error (RMSE): {calc_rms_stereo(objPoints_l, imgPoints_l, imgPoints_r, stereo_module_coeff.K1, stereo_module_coeff.D1, stereo_module_coeff.K2, stereo_module_coeff.D2, stereo_module_coeff.R, stereo_module_coeff.T)}")