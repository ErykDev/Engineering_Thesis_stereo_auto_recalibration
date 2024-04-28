import cv2
import argparse

from calibration_utils import *

parser = argparse.ArgumentParser(description='live_frame_collector')

parser.add_argument('--camera_ids', nargs='+', type=int, default=(2,4),
                    help='Array of cameras_ids')

parser.add_argument('--cameras_frame_shape', nargs='+', type=int, default=(1024, 768),
                    help='Expected frame shape w h')


args = parser.parse_args()

img_shape = args.cameras_frame_shape

def main():
    # define a video capture object
    left_cam = cv2.VideoCapture(args.camera_ids[0])
    right_cam = cv2.VideoCapture(args.camera_ids[1])

    setupCam(left_cam, img_shape[0], img_shape[1])
    setupCam(right_cam, img_shape[0], img_shape[1])

    while(True):
        left_cam.grab()
        _, frame_left = left_cam.read()

        frame = cv2.cvtColor(frame_left, cv2.COLOR_YUV2BGR_YUY2) 

        cv2.imshow('Preview left', frame)
        k = cv2.waitKey(1)
        if(k%256 == ord('q')):
            break
  
    # Destroy all the windows
    cv2.destroyAllWindows()
    while(True):
        right_cam.grab()
        #_, frame_left = left_cam.read()
        _, frame_right = right_cam.read()

        frame = cv2.cvtColor(frame_right, cv2.COLOR_YUV2BGR_YUY2) 

        cv2.imshow('Preview right', frame)
        k = cv2.waitKey(1)
        if(k%256 == ord('q')):
            break

    # Destroy all the windows
    cv2.destroyAllWindows()

    
    # After the loop release the cap object
    left_cam.release()
    right_cam.release()
    
    

if __name__ == '__main__':
    main()
