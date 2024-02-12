import socket
import cv2
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
from SuperGlue.models.matching import Matching
import torchvision.transforms as transforms
from client_utils import *
from datasets import __datasets__
from models import __models__
from SuperGlue.scoring_module import score_match
from SuperGlue.recalibration_module import recalculare_rotation
from utils import *
from torch.utils.data import DataLoader
from threading import Thread
from multiprocessing import Process
import matplotlib.pyplot as plt
import torch

# usage python3 client.py --host_ip 192.168.100.5

class PSMNetConnectionInteface(ConnectionInteface):
    @make_nograd_func
    
    def __init__(self):
        self.model = 'Fast_ACVNet_plus'
        self.maxdisp = 288
        self.cuda = False

        self.normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        self.infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**self.normal_mean_var)])
        
    def init_empty_network(self, cuda):
        self.model = __models__[self.model](self.maxdisp, False)
        self.cuda = cuda
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

        self.model.eval()
    
    def init_network(self, model_path, cuda):
        self.model = __models__[self.model](self.maxdisp, False)
        self.cuda = cuda

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
        
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model'])

        self.model.eval()
    
    def processImageSet(self, left_image, right_image):
        left_image = np.array(left_image)
        right_image = np.array(right_image)

        imgL = self.infer_transform(left_image)
        imgR = self.infer_transform(right_image)
        
        # Padding
        if imgL.shape[1] % 32 != 0:
            times = imgL.shape[1]//32       
            top_pad = (times+1)*32 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 32 != 0:
            times = imgL.shape[2]//32                       
            right_pad = (times+1)*32-imgL.shape[2]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        if self.cuda:
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            #print("CUDA")


        
        with torch.no_grad():
            #self.model.eval()
            disp = self.model(imgL, imgR)
        disp_est_np = tensor2numpy(disp[-1])
        disp_est_np = np.array(disp_est_np, dtype=np.float32)
        disp_est_np = disp_est_np[:,top_pad:]
        #print(disp_est_np.shape)
        #disp_est_uint = np.round(disp_est_np * 256).astype(np.uint16)
        
        return disp_est_np

def dispToDepth(disp, baseline, focal):
    disp_gt = disp
    depth_gt = np.zeros_like(disp_gt)
    idx = disp_gt != 0.0
    #print(depth_gt)
    #print(disp_gt)
    depth_gt[idx] = (baseline * focal) / disp_gt[idx]
    return depth_gt

parser = argparse.ArgumentParser(description='agx video server')

parser.add_argument('--host_ip', type=str,
                    help='ip of the server')

parser.add_argument('--pinhole_conf_path', type=str, default='./pinhole/',
                    help='pinhole (cam 2/3) coof')


parser.add_argument('--fisheye_conf_path', type=str, default='./fisheye/',
                    help='pinhole (cam 1/4) coof')

parser.add_argument('--host_port', type=int, default=9999,
                    help='ip of the server')

args = parser.parse_args()

DOWNSCALE_RATIO = 2.5


pinhole_set_coeff = load_camera_coeff(args.pinhole_conf_path, 'pinhole')

pinhole_set_coeff.K1 = rescale_camera_matrix(pinhole_set_coeff.K1, 1280, 1280/DOWNSCALE_RATIO)
pinhole_set_coeff.K2 = rescale_camera_matrix(pinhole_set_coeff.K2, 1280, 1280/DOWNSCALE_RATIO)

fisheye_set_coeff = load_camera_coeff(args.fisheye_conf_path, 'fisheye')

fisheye_set_coeff.K1 = rescale_camera_matrix(fisheye_set_coeff.K1, 1280, 1280/DOWNSCALE_RATIO)
fisheye_set_coeff.K2 = rescale_camera_matrix(fisheye_set_coeff.K2, 1280, 1280/DOWNSCALE_RATIO)


h, w, = int(720/ DOWNSCALE_RATIO), int(1280/DOWNSCALE_RATIO)

# Load the SuperPoint and SuperGlue models.
device = 'cpu'
print('Running inference on device \"{}\"'.format(device))

scoring = Matching({
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

matching = Matching({
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 500
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}).eval().to(device)
connIntef = PSMNetConnectionInteface()
connIntef.init_network('./checkpoint_000099.ckpt',cuda=True)
pixel_size = 0.003

sensor_size_x = w * pixel_size
sensor_size_y = h * pixel_size

mapx1_p, mapy1_p, mapx2_p, mapy2_p, P1_p = stereoRectifyInitUndistortRectifyMapPinhole(pinhole_set_coeff, (w, h))
mapx1_f, mapy1_f, mapx2_f, mapy2_f, P1_f = stereoRectifyInitUndistortRectifyMapFisheye(fisheye_set_coeff, (w, h))

fov_x, fov_y, focal_len, principal, aspect = \
    cv2.calibrationMatrixValues(P1_p[:3, :3], (w, h),
                                sensor_size_x, sensor_size_y)

focalv2 = focal_len / pixel_size

PACK_SIZE = 56320  #56320 #4096
DATA_SIZE = PACK_SIZE - 20

#Create a datagram socket
client_socket = socket.socket(socket.AF_INET, 
                              socket.SOCK_DGRAM, 
                              socket.IPPROTO_UDP)

# We are sending that so the server knows our adress/port
client_socket.sendto(b'Hello', (args.host_ip, args.host_port)) #establish connection

#time.sleep(2)

last_scores_p = []
last_scores_f = []

def evaluate_and_correct_camera_coeff(left_frame, right_frame, camera_coeff, size, last_scores, mapx1, mapy1, mapx2, mapy2):
    score = score_match(left_frame, right_frame, scoring, device, camera_coeff, size)

    print("score: " + str(score))

    last_scores.insert(0, score)

    recalibrate = len(last_scores) > 5 and Average(last_scores) > 5.0

    if recalibrate:
        p_avg = Average(last_scores)
        new_R = recalculare_rotation(left_frame, right_frame, matching, device, camera_coeff, size)
    
        new_score = score_match(left_frame, right_frame, scoring, device, device, CameraCoeff(camera_coeff.K1, camera_coeff.K2, camera_coeff.D1, camera_coeff.D2, new_R, camera_coeff.T, camera_coeff.Type), size)

        if new_score < p_avg:
            last_scores = []

            print('Updating ' + camera_coeff.Type + ' Coeff')

            if camera_coeff.Type == 'pinhole':
               pinhole_set_coeff.R =  new_R
               mapx1, mapy1, mapx2, mapy2, P1 = stereoRectifyInitUndistortRectifyMapPinhole(pinhole_set_coeff (w, h))

            if camera_coeff.Type == 'fisheye':
               fisheye_set_coeff.R =  new_R
               mapx1, mapy1, mapx2, mapy2, P1 = stereoRectifyInitUndistortRectifyMapFisheye(fisheye_set_coeff, (w, h))
    #else:
    #    print('No need to recalibrate ' + camera_coeff.Type)

    if(len(last_scores) >= 10):
        last_scores = last_scores[:-1]



frame_numb = 0
cmap = plt.cm.get_cmap('gist_rainbow')
MAX_DEPTH = 180
print('Reading data')
x = 1 # displays the frame rate every 1 second
counter = 0
start_time = time.time()
while(True):   
    

    frame_id = None

    # Waiting for starting packet containing numer of next packets
    while(True):
        data = client_socket.recv(PACK_SIZE)
        if(len(data) == 20): # info pack
            # Reding number of expected packets 

            frame_id = data[:10]
            parts = data[11]#int.from_bytes(data[11], 'little')

            break

    buffer = bytearray(parts * DATA_SIZE)

    collected_parts = 0

    # Reading expected packets
    for i in range(parts):
        data = client_socket.recv(PACK_SIZE)

        part_num = data[12]
        temp_frame_id = data[:10]

        if(frame_id == temp_frame_id):
            buffer[part_num * DATA_SIZE : (part_num * DATA_SIZE) + DATA_SIZE] = data[20:]
            collected_parts = collected_parts + 1

    rawData = np.frombuffer(buffer, dtype=np.uint8)
    frame = cv2.imdecode(rawData, cv2.IMREAD_COLOR)
    #print(frame.shape)
    if(collected_parts != parts):
        continue

    if(frame is None): # failed to create frame
        print('lost frame')
        continue

    frame_numb += 1


    l_frame_p = frame[:,:w]
    r_frame_p = frame[:, w:]

    # l_frame_f = frame[:h,:w]
    # r_frame_f = frame[:h, w:]
    #print("frame_numb " + str(frame_numb))

    # if (frame_numb % 250 == 0):
        # #print("starting recalibration process")
        # Thread(target = evaluate_and_correct_camera_coeff, args = (l_frame_p, r_frame_p, pinhole_set_coeff, (w, h), last_scores_p, mapx1_p, mapy1_p, mapx2_p, mapy2_p)).start()
        # #Thread(target = evaluate_and_correct_camera_coeff, args = (l_frame_f, r_frame_f, fisheye_set_coeff, (w, h), last_scores_f, mapx1_f, mapy1_f, mapx2_f, mapy2_f)).start()

    if(frame_numb > 300):
        frame_numb = 1
        #print(last_scores)
    
    # pinhole_frames
    l_frame_p = cv2.remap(l_frame_p, mapx1_p, mapy1_p, cv2.INTER_LINEAR)
    r_frame_p = cv2.remap(r_frame_p, mapx2_p, mapy2_p, cv2.INTER_LINEAR)
    print(l_frame_p.shape)
    #print(l_frame_p.shape)
    # fisheye_frames
    #l_frame_f = cv2.remap(l_frame_f, mapx1_f, mapy1_f, cv2.INTER_LINEAR)
    #r_frame_f = cv2.remap(r_frame_f, mapx2_f, mapy2_f, cv2.INTER_LINEAR)
    #l_frame_p = cv2.cvtColor(l_frame_g,cv2.COLOR_GRAY2RGB)
    #l_frame_p = cv2.cvtColor(l_frame_g,cv2.COLOR_GRAY2RGB)
    disp = np.squeeze(connIntef.processImageSet(l_frame_p, r_frame_p))
    
    # print(disp.shape)
    # print(focalv2)
    #skimage.io.imsave("test.png", np.squeeze(disp))
    # print(focalv2)
    depth = dispToDepth(disp,0.15,focalv2)
    
    # print(depth.shape)
    depth_visual = 255 * cmap((depth / MAX_DEPTH)) #cmap works on ranges 0-1
    depth_visual = depth_visual.astype(np.uint8) # opencv expects uint8 on flot image 
    depth_visual[depth < 0.2] = [0, 0, 0, 0] # removing to close objects 
    # print(depth_visual.shape)
    img_im_l_r = cv2.cvtColor(l_frame_p, cv2.COLOR_BGR2BGRA)
    # print(img_im_l_r.shape)
    depth_visual_p = cv2.cvtColor(depth_visual, cv2.COLOR_RGBA2BGRA)
    
    added_image = cv2.addWeighted(img_im_l_r,0.9,depth_visual,0.55,0)

    
    cv2.imshow("depth.png", added_image)
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    k = cv2.waitKey(1) # wait for key press
    if(k%256 == ord('q')):
        break
    if(k%256 == ord('s')):
        cv2.imwrite('file.jpg', frame)

