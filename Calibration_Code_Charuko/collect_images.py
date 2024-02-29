import cv2
import argparse
import os
import os.path
from os import path

from threading import Thread, Lock

from threading import Lock
import time

from calibration_utils import *

parser = argparse.ArgumentParser(description='live_frame_collector')


parser.add_argument('--save_path', default="./../",
                    help='path to save frames')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1600, 960),
                    help='Expected frame shape w h')

parser.add_argument('--collection_freq', type=int, default=5,
                    help='time in sec collection frequency')

args = parser.parse_args()

img_shape = args.cameras_frame_shape

camera_lock = Lock()
counter = 0
done = False

def display_thread(left_cam, right_cam, camera_lock):
    cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
    while(True):
        global done
        
        camera_lock.acquire()
        ret, frame_left = left_cam.read()
        ret1, frame_right = right_cam.read()

        camera_lock.release()
        
        #print(frame_left.shape)

        if(not ret or not ret1):
          continue

        frame = cv2.hconcat((frame_left, frame_right))
        frame = cv2.resize(frame, (int(frame.shape[1]/ 2), int(frame.shape[0]/ 2)))

        cv2.imshow('Preview', frame)
        k = cv2.waitKey(1)
        if(k%256 == ord('q')):
            done = True
            # Destroy all the windows
            cv2.destroyAllWindows()
            break

        

def saveing_thread(left_cam, right_cam, camera_lock, left_cam_save_path, right_cam_save_path):
    while(True):
        global counter
        global done

        time.sleep(args.collection_freq)

        camera_lock.acquire()
        ret, frame_left = left_cam.read()
        ret1, frame_right = right_cam.read()

        camera_lock.release()
        
        if(not ret or not ret1):
          continue

        counter = counter + 1

        cv2.imwrite(left_cam_save_path + "/" + str(counter) + ".png", frame_left)
        cv2.imwrite(right_cam_save_path + "/" + str(counter) + ".png", frame_right)

        if(done):
            break




def main():
    # define a video capture object
    left_cam = cv2.VideoCapture(args.camera_ids[0])
    right_cam = cv2.VideoCapture(args.camera_ids[1])

    setupCam(left_cam, img_shape[0], img_shape[1])
    setupCam(right_cam, img_shape[0], img_shape[1])

    left_cam_save_path   = args.save_path + '/cam' + str(int(args.camera_ids[0])) + '/'
    right_cam_save_path  = args.save_path + '/cam' + str(int(args.camera_ids[1])) + '/'

    if(not path.exists(left_cam_save_path)):
        os.mkdir(left_cam_save_path)

    if(not path.exists(right_cam_save_path)):
        os.mkdir(right_cam_save_path)


    # create threads
    t1 = Thread(target=display_thread, args=(left_cam, right_cam, camera_lock))
    t2 = Thread(target=saveing_thread, args=(left_cam, right_cam, camera_lock, left_cam_save_path, right_cam_save_path))

    # start the threads
    t1.start()
    t2.start()


    # wait for the threads to complete
    t1.join()
    t2.join()

    # After the loop release the cap object
    left_cam.release()
    right_cam.release()

if __name__ == '__main__':
    main()
