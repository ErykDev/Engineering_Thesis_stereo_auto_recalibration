import cv2
import numpy as np

class ConnectionInteface:
    # method for initializing Network
   
    def init_empty_network(cuda):
        raise Exception('Not implemented')
    
    def init_network(model_path, cuda):
        raise Exception('Not implemented')

    #Predicing single stereo set
    def processImageSet(left_image, right_image):
        raise Exception('Not implemented')
    
    #Redicing Batch of Stereo sets
    def processBatch(Batch):
        raise Exception('Not implemented')
    
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


def rescale_camera_matrix(camera_matrix, org_width, new_width):
    scale = new_width / org_width

    new_cam_matrix = camera_matrix * scale
    new_cam_matrix[2][2] = 1

    return new_cam_matrix


def stereo_rectify_map_fisheye(camera_coeff, size):
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,4))
    P2 = np.zeros(shape=(3,4))

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        camera_coeff.K1, camera_coeff.D1,
        camera_coeff.K2, camera_coeff.D2,
        size,
        camera_coeff.R,
        camera_coeff.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0
    )

    mapx1, mapy1 = cv2.fisheye.initUndistortRectifyMap(
        camera_coeff.K1, camera_coeff.D1,
        R1, P1,
        size,
        cv2.CV_16SC2
    )
    mapx2, mapy2 = cv2.fisheye.initUndistortRectifyMap(
        camera_coeff.K2, camera_coeff.D2,
        R2, P2,
        size,
        cv2.CV_16SC2
    )
    return mapx1, mapy1, mapx2, mapy2, P1, R1, P2, R2

def stereo_rectify_map(camera_coeff, size):
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,4))
    P2 = np.zeros(shape=(3,4))

    cv2.stereoRectify(camera_coeff.K1, 
                      camera_coeff.D1, 
                      camera_coeff.K2, 
                      camera_coeff.D2, 
                      size, 
                      camera_coeff.R, 
                      camera_coeff.T, 
                      R1, R2, P1, P2, alpha=0.0, 
                      flags=cv2.CALIB_ZERO_DISPARITY)

    mapx1, mapy1 = cv2.initUndistortRectifyMap(
        camera_coeff.K1, camera_coeff.D1,
        R1, P1,
        size,
        cv2.CV_32FC1
    )
    mapx2, mapy2 = cv2.initUndistortRectifyMap(
        camera_coeff.K2, camera_coeff.D2,
        R2, P2,
        size, 
        cv2.CV_32FC1
    )

    return mapx1, mapy1, mapx2, mapy2, P1, R1, P2, R2
