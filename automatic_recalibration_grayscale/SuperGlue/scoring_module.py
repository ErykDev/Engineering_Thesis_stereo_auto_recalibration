import numpy as np
import matplotlib.cm as cm

from SuperGlue.utils import make_matching_plot_fast


def score_match(mkpts0, mkpts1):
    diffs_y = np.abs(mkpts1[:, 1] - mkpts0[:, 1])
    return np.mean(diffs_y)

def draw_matches(image0, image1, kpts0, kpts1, mkpts0, mkpts1, mconf, matching_alg_name):
    text = [
        matching_alg_name,
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    
    color = cm.jet(mconf)

    return make_matching_plot_fast(image0=image0, image1=image1, 
                                  kpts0=kpts0, kpts1=kpts1, 
                                  mkpts0=mkpts0, mkpts1=mkpts1, 
                                  color=color, text=text)