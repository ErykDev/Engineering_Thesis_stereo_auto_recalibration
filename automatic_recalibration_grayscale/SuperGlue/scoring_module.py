import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from threading import Lock


from SuperGlue.utils import make_matching_plot_fast, resize_imgs_to_tensor

from client_utils import stereo_rectify_map


def score_match(mkpts0, mkpts1):
    diffs_y = np.abs(mkpts1[:, 1] - mkpts0[:, 1])
    return np.mean(diffs_y)


def draw_matches(org_left_img_gray, org_right_img_gray, model, device, camera_coeff, size):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereo_rectify_map(camera_coeff, size)

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

    mconf = conf[valid]

    print('saving image')

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
            ]
    
    color = cm.jet(mconf)

    out = make_matching_plot_fast(image0=image0, image1=image1, 
                                  kpts0=kpts0, kpts1=kpts1, 
                                  mkpts0=mkpts0, mkpts1=mkpts1, 
                                  color=color, text=text)

    cv2.imwrite("matches.png", out)