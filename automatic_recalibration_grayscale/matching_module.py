import cv2
import numpy as np
from SuperGlue.utils import resize_imgs_to_tensor
from client_utils import stereo_rectify_map


def get_matched_fetures_super_glue(org_left_img_gray, org_right_img_gray, model, device, camera_coeff, size, lock):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereo_rectify_map(camera_coeff, size)

    # Lock for accessing projection matrices
    lock.acquire()

    image0 = cv2.remap(image0, mapx1, mapy1, cv2.INTER_LINEAR)
    image1 = cv2.remap(image1, mapx2, mapy2, cv2.INTER_LINEAR)

    lock.release()

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
    mconf = conf[valid]

    print(conf)

    return kpts0, kpts1, mkpts0, mkpts1, conf, mconf



def get_matched_fetures(org_left_img_gray, org_right_img_gray, descriptor, matcher, camera_coeff, size, lock):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereo_rectify_map(camera_coeff, size)

    # Lock for accessing projection matrices
    lock.acquire()

    image0 = cv2.remap(image0, mapx1, mapy1, cv2.INTER_LINEAR)
    image1 = cv2.remap(image1, mapx2, mapy2, cv2.INTER_LINEAR)

    lock.release()

    image0 = cv2.resize(image0, size, interpolation= cv2.INTER_LINEAR) 
    image1 = cv2.resize(image1, size, interpolation= cv2.INTER_LINEAR) 


    kp1, des1 = descriptor.detectAndCompute(image0, None)
    kp2, des2 = descriptor.detectAndCompute(image1, None)

    matches = []

    if(des1 is not None and des2 is not None):
        # Find the two nearest neighbors for each descriptor
        matches = matcher.knnMatch(des1, des2, k=2)

    # Apply the ratio test to filter out false matches
    mkpts0 = []
    mkpts1 = []
    for m, n in matches:
        #print(m.distance)
        mkpts0.append(list(kp1[m.queryIdx].pt))
        mkpts1.append(list(kp2[m.trainIdx].pt))

    # Convert keypoints to arrays of (x, y) coordinates
    kpts0 = [list(kp.pt) for kp in kp1]
    kpts1 = [list(kp.pt) for kp in kp2]

    return np.array(kpts0), np.array(kpts1), np.array(mkpts0), np.array(mkpts1), np.array([0.0 for i in kpts0]), np.array([0.0 for i in mkpts0])
