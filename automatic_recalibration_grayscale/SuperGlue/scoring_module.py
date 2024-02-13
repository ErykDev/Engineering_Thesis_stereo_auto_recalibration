import torch
import cv2
import numpy as np


from SuperGlue.utils import resize_imgs_to_tensor

from client_utils import stereoRectifyInitUndistortRectifyMapPinhole

torch.set_grad_enabled(False)



def score_match(org_left_img_gray, org_right_img_gray, model, device, camera_coeff, size):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereoRectifyInitUndistortRectifyMapPinhole(camera_coeff, size)

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

    diffs_y = []
    
    for point1, point2 in zip(mkpts0, mkpts1):
        y1 = point1[1]
        y2 = point2[1]
        diffs_y.append(abs(y2- y1))

    return sum(diffs_y) / len(diffs_y)


def draw_matches(org_left_img_gray, org_right_img_gray, model, device, camera_coeff, size):
    image0 = org_left_img_gray
    image1 = org_right_img_gray


    mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereoRectifyInitUndistortRectifyMapPinhole(camera_coeff, size)


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

    imMatches = cv2.drawMatches(image0, mkpts0, image1, mkpts1, matches, None)
    cv2.imwrite("matches.png", imMatches)