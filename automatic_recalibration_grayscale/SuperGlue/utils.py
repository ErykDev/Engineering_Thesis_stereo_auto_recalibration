import numpy as np
import torch

import cv2
import math


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def resize_imgs_to_tensor(image, device, resize, rotation, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def read_image(path, device, resize, rotation, resize_float, K, K_optimal, D):
    image = read_UYVY_bgr(path, 1920, 1200) #cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #image = undistort_pinhole_image(image, K, K_optimal, D)

    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

def angle_bettwen_points(x1, x2, y1, y2):
    angle_rad = math.atan2(abs(y2-y1), abs(x2-x1))

    return math.degrees(angle_rad)

def Average(lst):
    return sum(lst) / len(lst)


def read_UYVY_grayscale(path, width, height):
    with open(path, "rb") as src_file:
        raw_data = np.fromfile(src_file, dtype=np.uint8, count=width*height*2)
        return raw_data.reshape(height, width, 2)[0]

def read_UYVY_bgr(path, width, height):
    with open(path, "rb") as src_file:
        raw_data = np.fromfile(src_file, dtype=np.uint8, count=width*height*2)
        im = raw_data.reshape(height, width, 2)
        return cv2.cvtColor(im, cv2.COLOR_YUV2BGR_Y422)

def get_optimal_camera_matrix(K, Dist, img_width, img_height):
    K1, _ = cv2.getOptimalNewCameraMatrix(
	K, Dist, (img_width, img_height), 0, (img_width, img_height))
	
    return K1

def undistort_pinhole_image(image, camera_matrix, optimal_cameramatrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, optimal_cameramatrix)
