import cv2
import argparse
import copy

from SuperGlue.models.matching import Matching
from client_utils import *
from SuperGlue.scoring_module import score_match, draw_matches

from SuperGlue.recalibration_module import calculate_rotation
from matching_module import get_matched_fetures_super_glue

from utils import *
from threading import Thread, Lock


import os


scene_path = '/home/eryk-dev/Desktop/Engineering_Thesis_stereo_auto_recalibration/Collected_Scenes/scena2/'

starting_calibration_path = scene_path + 'Starting_Calibration/'
starting_calibration_path = starting_calibration_path + os.listdir(starting_calibration_path)[0] + '/'


#ending_calibration_path = scene_path + 'ENDING_Calibration/'
#ending_calibration_path = ending_calibration_path + os.listdir(ending_calibration_path)[0] + '/'


collected_frames_path = scene_path + 'Frames/'


DOWNSCALE_RATIO = 1

stereo_module_coeff = load_camera_coeff(starting_calibration_path, 'pinhole')

original_width = 1024
original_height = 768

stereo_module_coeff.K1 = rescale_camera_matrix(stereo_module_coeff.K1, 
                                               original_width, 
                                               original_width/DOWNSCALE_RATIO)
stereo_module_coeff.K2 = rescale_camera_matrix(stereo_module_coeff.K2, 
                                               original_width, 
                                               original_width/DOWNSCALE_RATIO)

h, w, = int(original_height/DOWNSCALE_RATIO), int(original_width/DOWNSCALE_RATIO)

# Load the SuperPoint and SuperGlue models.
device = 'cpu'
print('Running inference on device \"{}\"'.format(device))

scorer = Matching({
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 200
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)


matcher = Matching({
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 300
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)


scores = []
fovs = []

evaluation_interval = 2

lock = Lock()
counter = Max_Counter(start=1, max=evaluation_interval + 1)
last_scores = Scores(max_length=3)

fps = FPS(pooling_size=3)

frame_num = 0
#frame_fov = []


def evaluate_and_correct_camera_coeff(left_frame, right_frame, size, lock):
    global mapx1, mapy1, mapx2, mapy2, stereo_module_coeff, last_scores, frame_num#, frame_fov


    _, _, mkpts0, mkpts1, _, _ = get_matched_fetures_super_glue(left_frame, 
                                                     right_frame, 
                                                     scorer, 
                                                     device, 
                                                     stereo_module_coeff, 
                                                     size, 
                                                     lock)

    score = score_match(mkpts0, mkpts1)

    scores.append(score)

    print("score: " + str(score))
    print("last_scores len" + str(len(last_scores)))

    last_scores.append(score)

    if len(last_scores) >= 2 and last_scores.average() > 1.5:

        _, _, mkpts0, mkpts1, _, _ = get_matched_fetures_super_glue(left_frame, 
                                                     right_frame, 
                                                     matcher, 
                                                     device, 
                                                     stereo_module_coeff, 
                                                     size, 
                                                     lock)
        
        R = calculate_rotation(mkpts0, mkpts1, stereo_module_coeff)

        copy_stereo_module_coeff = copy.deepcopy(stereo_module_coeff)
        copy_stereo_module_coeff.R = R


        _, _, mkpts0, mkpts1, _, _ = get_matched_fetures_super_glue(left_frame, 
                                                         right_frame, 
                                                         scorer, 
                                                         device, 
                                                         copy_stereo_module_coeff, 
                                                         size, 
                                                         lock)

        new_score = score_match(mkpts0, mkpts1)
        
        if new_score < last_scores.average():
            last_scores.reset()

            print('Updating Projection Matrices')


            # Lock for updatating projection matrices
            lock.acquire()

            stereo_module_coeff.R = R
            mapx1, mapy1, mapx2, mapy2, P1, R1, P2, R2 = stereo_rectify_map(stereo_module_coeff, (w, h))

            pixel_size = 0.003
            image_pixel_size_x = w
            image_pixel_size_y = h

            imageSize=(image_pixel_size_x, image_pixel_size_y)

            sensor_size_x = image_pixel_size_x * pixel_size
            sensor_size_y = image_pixel_size_y * pixel_size

            #fov_x, fov_y, focal_len, principal, aspect = \
            #cv2.calibrationMatrixValues(P1[:3, :3], (image_pixel_size_x, image_pixel_size_y),
            #                            sensor_size_x, sensor_size_y)  
            

            #frame_fov.append((frame_num, fov_x))

            lock.release()
    #else:
    #    print('No need to recalibrate ' + camera_coeff.Type)


if __name__ == "__main__":
    mapx1, mapy1, mapx2, mapy2, P1, R1, P2, R2 = stereo_rectify_map(stereo_module_coeff, (w, h))

    pixel_size = 0.003
    image_pixel_size_x = w
    image_pixel_size_y = h

    imageSize=(image_pixel_size_x, image_pixel_size_y)

    sensor_size_x = image_pixel_size_x * pixel_size
    sensor_size_y = image_pixel_size_y * pixel_size

    fov_x, fov_y, focal_len, principal, aspect = \
    cv2.calibrationMatrixValues(P1[:3, :3], (image_pixel_size_x, image_pixel_size_y),
                                sensor_size_x, sensor_size_y)  
    

    #frame_fov.append((0, fov_x))


    frames_left = os.listdir(collected_frames_path + '/cam2/')
    frames_left.sort(key=lambda x: int(x.split('.')[0]))

    frames_right = os.listdir(collected_frames_path + '/cam4/')
    frames_right.sort(key=lambda x: int(x.split('.')[0]))

    #Main loop
    for left_path, right_path in zip(frames_left, frames_right):   
        
        counter.increment()
        frame_num = frame_num + 1

        frame_left = cv2.imread(collected_frames_path + 'cam2/'+ left_path)
        frame_right = cv2.imread(collected_frames_path + 'cam4/'+ right_path)

        org_frame_left = copy.deepcopy(frame_left)
        org_frame_right = copy.deepcopy(frame_right)
            
        #Downscale images
        frame_left = cv2.resize(frame_left, (w, h))
        frame_right = cv2.resize(frame_right, (w, h))

        #BGR to Grayscale
        frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


        if (counter.count == evaluation_interval):
            print("starting recalibration process")
            Thread(target = evaluate_and_correct_camera_coeff, 
                args =(frame_left_gray, 
                frame_right_gray, 
                (w, h), 
                lock)).start()

        # Lock for accesing projection matrices
        lock.acquire()
        
        # pinhole_frames
        frame_left_remaped = cv2.remap(frame_left, mapx1, mapy1, cv2.INTER_LINEAR)
        frame_right_remaped = cv2.remap(frame_right, mapx2, mapy2, cv2.INTER_LINEAR)

        frame_left_gray_remaped = cv2.cvtColor(frame_left_remaped, cv2.COLOR_BGR2GRAY)
        frame_right_gray_remaped = cv2.cvtColor(frame_right_remaped, cv2.COLOR_BGR2GRAY)

        lock.release()

        fps.update()
        print("FPS: ", fps.calculate())

        frame_left_gray = np.squeeze(frame_left_gray)
        frame_right_gray = np.squeeze(frame_right_gray)

        kpts0, kpts1, mkpts0, mkpts1, conf, mconf = get_matched_fetures_super_glue(frame_left_gray, 
                                                            frame_right_gray, 
                                                            matcher, 
                                                            device, 
                                                            stereo_module_coeff, 
                                                            (w, h), 
                                                            lock)
        
        frame_left_gray_remaped = np.squeeze(frame_left_gray_remaped)
        frame_right_gray_remaped = np.squeeze(frame_right_gray_remaped)

        frame = draw_matches(frame_left_gray_remaped, frame_right_gray_remaped, kpts0, kpts1, mkpts0, mkpts1, mconf, '')

        #print(frame.shape)


        #frame = cv2.hconcat((frame_left_remaped, frame_right_remaped))
        #frame = cv2.resize(frame, (int(frame.shape[1]/ 1), int(frame.shape[0]/ 1)))

        # Display the resulting frame
        cv2.imshow('View Display',  frame)

        k = cv2.waitKey(1) # wait for key press
        if(k%256 == ord('q')):
            break
        if(k%256 == ord('m')):
            kpts0, kpts1, mkpts0, mkpts1, conf, mconf = get_matched_fetures_super_glue(frame_left_gray, 
                                                            frame_right_gray, 
                                                            matcher, 
                                                            device, 
                                                            stereo_module_coeff, 
                                                            (w, h), 
                                                            lock)
            
            out = draw_matches(frame_left_remaped, frame_right_remaped, kpts0, kpts1, mkpts0, mkpts1, mconf, 'SuperGlue')
            cv2.imwrite("matches.png", out)

    print(scores)

    np.savetxt('R.txt', stereo_module_coeff.R)

    #for i, frame_fov in enumerate(frame_fov):
    #    print(frame_fov)
