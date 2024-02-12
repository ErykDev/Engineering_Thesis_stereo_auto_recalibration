import numpy as np
import torch

import cv2

from client_utils import stereoRectifyInitUndistortRectifyMapPinhole, stereoRectifyInitUndistortRectifyMapFisheye

from SuperGlue.utils import resize_imgs_to_tensor

torch.set_grad_enabled(False)


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#w, h = 1920, 1200
def recalculare_rotation(org_left_img_gray, org_right_img_gray, model, device, camera_coeff, size):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    if camera_coeff.Type == 'pinhole':
        mapx1, mapy1, mapx2, mapy2 = stereoRectifyInitUndistortRectifyMapPinhole(camera_coeff, size)

    if camera_coeff.Type == 'fisheye':
        mapx1, mapy1, mapx2, mapy2 = stereoRectifyInitUndistortRectifyMapFisheye(camera_coeff, size)


    image0 = cv2.remap(image0, mapx1, mapy1, cv2.INTER_LINEAR)
    image1 = cv2.remap(image1, mapx2, mapy2, cv2.INTER_LINEAR)

    image0, inp0, scales0 = resize_imgs_to_tensor(image0, device, size, 0, False)
    image1, inp1, scales1 = resize_imgs_to_tensor(image1, device, size, 0, False)

    # Perform the matching.
    pred = model({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    #mconf = conf[valid]

    # Calculate rotation matrix from matches
    homography, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(homography, camera_coeff.K1)
    
    assert num != 0

    #print('recalculated rotation')
    #print(f'rool {rool} pich {pitch} yaw {yaw}')

    R_rec = np.dot(camera_coeff.R, Rs[0])

    assert isRotationMatrix(R_rec)

    return R_rec

