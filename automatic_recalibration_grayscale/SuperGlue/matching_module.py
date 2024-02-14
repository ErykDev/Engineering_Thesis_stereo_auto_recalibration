import cv2
from automatic_recalibration_grayscale.SuperGlue.utils import resize_imgs_to_tensor
from automatic_recalibration_grayscale.client_utils import stereo_rectify_map


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

    return kpts0, kpts1, mkpts0, mkpts1, conf, mconf