#!/bin/bash   
echo "Give a scenne name:"    
read NEW_SCENNE  
ORIG_DIR=$(pwd)  

mkdir -p $ORIG_DIR/Collected_Scenes;

MAIN_Dir=$ORIG_DIR/Collected_Scenes/$NEW_SCENNE
mkdir -p $MAIN_Dir;

STARTING_CALIB_PATH=$MAIN_Dir/Starting_Calibration
mkdir -p $STARTING_CALIB_PATH;


#echo "Starting collection of strarting callibration frames" 
#python3 ./Calibration_Code_Charuko/live_stereo_calibration_collector.py --save_path $STARTING_CALIB_PATH/


COLLECTED_FRAMES_PATH=$MAIN_Dir/Frames
mkdir -p $COLLECTED_FRAMES_PATH

echo "Starting Frame collection" 
python3 ./Calibration_Code_Charuko/collect_images.py --save_path $COLLECTED_FRAMES_PATH/

ENDING_CALIB_PATH=$MAIN_Dir/ENDING_Calibration
mkdir -p $ENDING_CALIB_PATH;

echo "Starting collection of end callibration frames" 
python3 ./Calibration_Code_Charuko/live_stereo_calibration_collector.py --save_path $ENDING_CALIB_PATH/