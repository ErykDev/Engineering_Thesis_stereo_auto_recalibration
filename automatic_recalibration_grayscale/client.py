import cv2
import argparse
import time

from SuperGlue.models.matching import Matching
from client_utils import *
from datasets import __datasets__
from models import __models__
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
        Counter.__init__(start=start)
        self.max = max

    def increment(self):
        Counter.increment()
        
        if self.count >= self.max:
            self.reset()

   
class FPS(Counter):
    def __init__(self, start_time = time.time()):
        self.start_time = start_time
        Counter.__init__(start=0)

    def update(self):
        Counter.increment()

    def calculate(self):
        return self.count / (time.time() - self.start_time)
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

parser.add_argument('--pinhole_conf_path', type=str, default='./pinhole/',
                    help='pinhole (cam 2/3) coof')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1600, 960),
                    help='Expected frame shape w h')


args = parser.parse_args()

DOWNSCALE_RATIO = 2

pinhole_set_coeff = load_camera_coeff(args.pinhole_conf_path, 'pinhole')

original_width = args.cameras_frame_shape[0]

pinhole_set_coeff.K1 = rescale_camera_matrix(pinhole_set_coeff.K1, original_width, original_width/DOWNSCALE_RATIO)
pinhole_set_coeff.K2 = rescale_camera_matrix(pinhole_set_coeff.K2, original_width, original_width/DOWNSCALE_RATIO)

h, w, = int(args.cameras_frame_shape[1]/ DOWNSCALE_RATIO), int(original_width/DOWNSCALE_RATIO)

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
    global last_scores, mapx1, mapy1, mapx2, mapy2, pinhole_set_coeff

    score = score_match(left_frame, right_frame, matcher, device, pinhole_set_coeff, size)

    print("score: " + str(score))

    last_scores.append(score)

    recalibrate = len(last_scores) == 10 and last_scores.average() > 5.0

    if recalibrate:
        new_R = recalculare_rotation(left_frame, right_frame, matcher, device, pinhole_set_coeff, size)
    
        new_score = score_match(left_frame, right_frame, matcher, device, device, 
            CameraCoeff(pinhole_set_coeff.K1, pinhole_set_coeff.K2, pinhole_set_coeff.D1, 
                        pinhole_set_coeff.D2, new_R, pinhole_set_coeff.T, pinhole_set_coeff.Type), size)

        if new_score < last_scores.average():
            last_scores.reset()

            print('Updating ' + pinhole_set_coeff.Type + ' Coeff')


            # Lock for updatating projection matrices
            lock.acquire()

            pinhole_set_coeff.R =  new_R
            mapx1, mapy1, mapx2, mapy2, _,_,_,_ = stereoRectifyInitUndistortRectifyMapPinhole(pinhole_set_coeff (w, h))

            lock.release()
    #else:
    #    print('No need to recalibrate ' + camera_coeff.Type)

mapx1_p, mapy1_p, mapx2_p, mapy2_p, _, _, _, _ = stereoRectifyInitUndistortRectifyMapPinhole(pinhole_set_coeff, (w, h))

left_cam = cv2.VideoCapture(args.camera_ids[0])
right_cam = cv2.VideoCapture(args.camera_ids[1])

setupCam(left_cam, args.cameras_frame_shape[0],  args.cameras_frame_shape[1])
setupCam(right_cam, args.cameras_frame_shape[0],  args.cameras_frame_shape[1])

evaluation_interval = 500

lock = Lock()
counter = Max_Counter(start=1, max=evaluation_interval + 1)
last_scores = Scores(max_length=10)

fps = FPS(start_time=time.time())

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
        Thread(target = evaluate_and_correct_camera_coeff, args =(l_gray_frame, r_gray_frame, pinhole_set_coeff, (w, h), lock)).start()


    
    # Lock for accesing projection matrices
    lock.acquire()
    
    # pinhole_frames
    l_frame_remaped = cv2.remap(frame_left, mapx1_p, mapy1_p, cv2.INTER_LINEAR)
    r_frame_remaped = cv2.remap(frame_right, mapx2_p, mapy2_p, cv2.INTER_LINEAR)

    lock.release()



    fps.update()
    print("FPS: ", fps.calculate())

    k = cv2.waitKey(1) # wait for key press
    if(k%256 == ord('q')):
        break
    if(k%256 == ord('m')):
        draw_matches(l_gray_frame, r_gray_frame, matcher, device, pinhole_set_coeff, (w, h))
