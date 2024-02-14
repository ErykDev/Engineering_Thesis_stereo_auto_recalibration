import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from threading import Lock


from SuperGlue.utils import make_matching_plot_fast

from client_utils import stereo_rectify_map


def score_match(mkpts0, mkpts1):
    diffs_y = np.abs(mkpts1[:, 1] - mkpts0[:, 1])
    return np.mean(diffs_y)


def draw_matches(org_left_img_gray, org_right_img_gray, kpts0, kpts1, mkpts0, mkpts1, mconf):
    image0 = org_left_img_gray
    image1 = org_right_img_gray

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
            ]
    
    color = cm.jet(mconf)

    return make_matching_plot_fast(image0=image0, image1=image1, 
                                  kpts0=kpts0, kpts1=kpts1, 
                                  mkpts0=mkpts0, mkpts1=mkpts1, 
                                  color=color, text=text)

    #cv2.imwrite("matches.png", out)