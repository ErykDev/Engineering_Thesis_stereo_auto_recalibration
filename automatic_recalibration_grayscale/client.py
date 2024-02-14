import cv2
import argparse
import time

import copy

from SuperGlue.models.matching import Matching
from client_utils import *
from SuperGlue.scoring_module import score_match, draw_matches
from SuperGlue.recalibration_module import recalculare_rotation
from utils import *
from threading import Thread, Lock



class Counter:
    def __init__(self, start = 1):
        self.start = start
        self.count = self.start

    def increment(self):
        self.count = self.count + 1
      
    def reset(self):
        self.count = 0

class Max_Counter(Counter):
    def __init__(self, start = 1, max = 300):
        Counter.__init__(self, start=start)
        self.max = max

    def increment(self):
        super().increment()
        
        if self.count >= self.max:
            self.reset()

   
class FPS():
    def __init__(self, pooling_size = 200):
        self.scores = Scores(pooling_size)
        self.start_time = time.time()
        
    def update(self):
        self.scores.append(1./(time.time() - self.start_time))
        self.reset()

    def reset(self):
        self.start_time = time.time()

    def calculate(self):
        return self.scores.average()
        #print("FPS: ", counter / (time.time() - start_time))

class Scores:
    def __init__(self, max_length= 10):
        self.scores = []
        self.max_length = max_length

    def append(self, score):
        self.scores.insert(0, score)

        if(len(self.scores) >= self.max_length):
            self.scores = self.scores[:-1]

    def reset(self):
        self.scores = []
    
    def average(self):
        return sum(self.scores) / len(self.scores)
    
    def __len__(self):
        return len(self.scores)

def setupCam(cam, w, h):
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    print('setting camera')

    time.sleep(1)

    print('setting resolution')
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)

    time.sleep(1)

    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(1)

    print('setting fps speed')
    cam.set(cv2.CAP_PROP_FPS, 30.000)


parser = argparse.ArgumentParser(description='agx video server')

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
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)


def evaluate_and_correct_camera_coeff(left_frame, right_frame, size, lock):
    global last_scores, mapx1, mapy1, mapx2, mapy2, stereo_module_coeff

    score = score_match(left_frame, right_frame, matcher, device, stereo_module_coeff, size)

    print("score: " + str(score))

    last_scores.append(score)

    recalibrate = len(last_scores) == 10 and last_scores.average() > 5.0

    if recalibrate:
        new_R = recalculare_rotation(left_frame, right_frame, matcher, device, stereo_module_coeff, size)

        copy_stereo_module_coeff = copy.deepcopy(stereo_module_coeff)
        copy_stereo_module_coeff.R = new_R
    
        new_score = score_match(left_frame, right_frame, matcher, device, device, copy_stereo_module_coeff, size)

        if new_score < last_scores.average():
            last_scores.reset()

            print('Updating ' + stereo_module_coeff.Type + ' Coeff')


            # Lock for updatating projection matrices
            lock.acquire()

            stereo_module_coeff.R =  new_R
            mapx1, mapy1, mapx2, mapy2, _,_,_,_ = stereo_rectify_map(stereo_module_coeff (w, h))

            lock.release()
    #else:
    #    print('No need to recalibrate ' + camera_coeff.Type)

mapx1_p, mapy1_p, mapx2_p, mapy2_p, _, _, _, _ = stereo_rectify_map(stereo_module_coeff, (w, h))

left_cam = cv2.VideoCapture(args.camera_ids[0])
right_cam = cv2.VideoCapture(args.camera_ids[1])

setupCam(left_cam,  original_width, original_height)
setupCam(right_cam, original_width, original_height)

evaluation_interval = 500

lock = Lock()
counter = Max_Counter(start=1, max=evaluation_interval + 1)
last_scores = Scores(max_length=10)

fps = FPS(pooling_size=200)

while(True):   
    
    counter.increment()

    ret, frame_left = left_cam.read()
    ret1, frame_right = right_cam.read()
        
    if(not ret or not ret1):
      continue


    #Downscale images
    frame_left = cv2.resize(frame_left, (w, h))
    frame_right = cv2.resize(frame_right, (w, h))

    
    #BGR to Grayscale
    l_gray_frame = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    r_gray_frame = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


    if (counter.count == evaluation_interval):
        print("starting recalibration process")
        Thread(target = evaluate_and_correct_camera_coeff, args =(l_gray_frame, r_gray_frame, stereo_module_coeff, (w, h), lock)).start()


    # Lock for accesing projection matrices
    lock.acquire()
    
    # pinhole_frames
    l_frame_remaped = cv2.remap(frame_left, mapx1_p, mapy1_p, cv2.INTER_LINEAR)
    r_frame_remaped = cv2.remap(frame_right, mapx2_p, mapy2_p, cv2.INTER_LINEAR)

    lock.release()


    fps.update()
    print("FPS: ", fps.calculate())
                               
    frame = cv2.hconcat((l_frame_remaped, r_frame_remaped))
    #print(frame.shape)
    #frame = cv2.resize(frame, (int(frame.shape[1]/ 1.8), int(frame.shape[0]/ 1.8)))

    # Display the resulting frame
    cv2.imshow('View Display',  frame)


    k = cv2.waitKey(1) # wait for key press
    if(k%256 == ord('q')):
        break
    if(k%256 == ord('m')):
        draw_matches(l_gray_frame, r_gray_frame, matcher, device, stereo_module_coeff, (w, h))
