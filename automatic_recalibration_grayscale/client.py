import cv2
import argparse
import copy

from SuperGlue.models.matching import Matching
from client_utils import *
from SuperGlue.scoring_module import score_match, draw_matches

from SuperGlue.recalibration_module import calculate_rotation
from SuperGlue.matching_module import get_matched_fetures_super_glue

from utils import *
from threading import Thread, Lock


parser = argparse.ArgumentParser(description='camera client')

parser.add_argument('--pinhole_conf_path', type=str, default='./../stereo_kalibracja_13_02_2024/',
                    help='pinhole (cam 2/3) coof')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1600, 960),
                    help='Expected frame shape w h')


args = parser.parse_args()

DOWNSCALE_RATIO = 2

stereo_module_coeff = load_camera_coeff(args.pinhole_conf_path, 'pinhole')

original_width = args.cameras_frame_shape[0]
original_height = args.cameras_frame_shape[1]

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

matcher = Matching({
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 200
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)


def evaluate_and_correct_camera_coeff(left_frame, right_frame, size, lock):
    global last_scores, mapx1, mapy1, mapx2, mapy2, stereo_module_coeff


    _, _, mkpts0, mkpts1, _, _ = get_matched_fetures_super_glue(left_frame, 
                                                     right_frame, 
                                                     matcher, 
                                                     device, 
                                                     stereo_module_coeff, 
                                                     size, 
                                                     lock)

    score = score_match(mkpts0, mkpts1)

    print("score: " + str(score))

    last_scores.append(score)

    if len(last_scores) == 10 and last_scores.average() > 5.0:
        R = calculate_rotation(left_frame, right_frame, matcher, device, stereo_module_coeff, size)

        copy_stereo_module_coeff = copy.deepcopy(stereo_module_coeff)
        copy_stereo_module_coeff.R = R


        _, _, mkpts0, mkpts1, _, _ = get_matched_fetures_super_glue(left_frame, 
                                                         right_frame, 
                                                         matcher, 
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
            mapx1, mapy1, mapx2, mapy2, _,_,_,_ = stereo_rectify_map(stereo_module_coeff (w, h))

            lock.release()
    #else:
    #    print('No need to recalibrate ' + camera_coeff.Type)

mapx1, mapy1, mapx2, mapy2, _, _, _, _ = stereo_rectify_map(stereo_module_coeff, (w, h))

left_cam = cv2.VideoCapture(args.camera_ids[0])
right_cam = cv2.VideoCapture(args.camera_ids[1])

setupCam(left_cam,  original_width, original_height)
setupCam(right_cam, original_width, original_height)

evaluation_interval = 500

lock = Lock()
counter = Max_Counter(start=1, max=evaluation_interval + 1)
last_scores = Scores(max_length=10)

fps = FPS(pooling_size=200)

#Main loop
while(True):   
    
    counter.increment()

    ret,  frame_left = left_cam.read()
    ret1, frame_right = right_cam.read()
        
    if(not ret or not ret1):
      continue


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
                      stereo_module_coeff, 
                      (w, h), 
                      lock)).start()


    # Lock for accesing projection matrices
    lock.acquire()
    
    # pinhole_frames
    frame_left_remaped = cv2.remap(frame_left, mapx1, mapy1, cv2.INTER_LINEAR)
    frame_right_remaped = cv2.remap(frame_right, mapx2, mapy2, cv2.INTER_LINEAR)

    lock.release()


    fps.update()
    print("FPS: ", fps.calculate())
                               
    frame = cv2.hconcat((frame_left_remaped, frame_right_remaped))
    #frame = cv2.resize(frame, (int(frame.shape[1]/ 1.8), int(frame.shape[0]/ 1.8)))

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