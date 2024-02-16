import cv2
import argparse
import os
import os.path
from os import path
from datetime import date

from calibration_utils import *

parser = argparse.ArgumentParser(description='live_frame_collector')


parser.add_argument('--save_path', default="./../",
                    help='path to save frames')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1600, 960),
                    help='Expected frame shape w h')

args = parser.parse_args()

img_shape = args.cameras_frame_shape


def main():
    # define a video capture object
    left_cam = cv2.VideoCapture(args.camera_ids[0])
    right_cam = cv2.VideoCapture(args.camera_ids[1])

    setupCam(left_cam, img_shape[0], img_shape[1])
    setupCam(right_cam, img_shape[0], img_shape[1])

    collected_frames = 0


    #kalib_folder = args.save_path

    #if(not path.exists(kalib_folder)):
    #    os.mkdir(kalib_folder)

    left_cam_save_path   = args.save_path + '/cam' + str(int(args.camera_ids[0])) + '/'
    right_cam_save_path  = args.save_path + '/cam' + str(int(args.camera_ids[1])) + '/'


    if(not path.exists(left_cam_save_path)):
        os.mkdir(left_cam_save_path)

    if(not path.exists(right_cam_save_path)):
        os.mkdir(right_cam_save_path)

    # Main loop
    while(True):
        # Capture the video frame
        # by frame
        ret, frame_left = left_cam.read()
        ret1, frame_right = right_cam.read()
        
        if(not ret or not ret1):
          continue
        

        #Saving frames
        cv2.imwrite(left_cam_save_path + "/" + str(collected_frames) + ".png", frame_left)
        cv2.imwrite(right_cam_save_path + "/" + str(collected_frames) + ".png", frame_right)

        collected_frames = collected_frames+1

        frame = cv2.hconcat((frame_left, frame_right))

        #print(frame.shape)

        frame = cv2.resize(frame, (int(frame.shape[1]/ 2), int(frame.shape[0]/ 2)))

        # Display the resulting frame
        cv2.imshow('View Display',  frame)

        k = cv2.waitKey(1) # wait for key press
      
        if(k%256 == ord('q')):
            break

    # After the loop release the cap object
    left_cam.release()
    right_cam.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
