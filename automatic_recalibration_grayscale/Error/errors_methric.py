from math import inf
import os
from time import sleep
import numpy as np
import gc
from matplotlib import pyplot as plt
from plotter_2 import *
from PIL import Image
from sum_mean import sum_mean,sum_mean_excel
import shutil
import cv2

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def load_path(list_filename):
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]
    right_images = [x[1] for x in splits]
    if len(splits[0]) == 2:  # ground truth not available
        return left_images, right_images, None
    else:
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images
def load_image(filename):
    return Image.open(filename).convert('RGB')

def load_disp(filename):
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) /256.
    return data

def load_depth(filename):
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) / 256.
    return data

def load_baseline(baseline_file):
    lines = read_all_lines(baseline_file)
    splits = [line.split() for line in lines]
    name= [x[0] for x in splits]
    bs= [x[1] for x in splits]

    return {"name":name,
            "bs":bs}
def errors_methric( ErrorsDirectory,list_filenames, predict_dir, fov, datapath):

        left_filenames, right_filenames, disp_filename=load_path(list_filenames)

        #baselines=load_baseline(baseline_file)

        for index in range(len(left_filenames)):
                        name=left_filenames[index].split("/")[-1]
                        #baseline=0.1511274634580715
                        #test_files=open("./test")
                        #test_names=test_files.readlines()
                        #test_files.close()
                        #for bs in test_names:
                        #    if name==bs.split(" ")[0]:
                        #        baseline=float(bs.split(" ")[1])
                        # for n, bs in zip (baselines["name"], baselines["bs"]):
                            # if n==left_filenames[index].split("/")[-4]:
                                # baseline=float(bs)
                                # print(baseline)
                        baseline= 45.37
                        predict_path= os.path.join(predict_dir, left_filenames[index].split('/')[-4]+left_filenames[index].split('/')[-3]+left_filenames[index].split('/')[-1])
                        ref_depth=os.path.join(datapath,left_filenames[index].split('/')[-4],left_filenames[index].split('/')[-3],"Depth_cam1_downscaled_filtered",name)
                        #predict_path=os.path.join(datapath,left_filenames[index].split('/')[-4],left_filenames[index].split('/')[-3],"Disp_cam2_downscaled_filtered",name)
                        #print(predict_path)
                        depthimg_norm=load_depth(ref_depth)
                        est_disp=load_disp(predict_path)
                        ref_disp=load_disp(os.path.join(datapath,disp_filename[index]))
                        imgL=load_image(os.path.join(datapath, left_filenames[index]))
                        imgR=load_image(os.path.join(datapath, right_filenames[index]))
                        #ref_disp=ref_disp/2.
                        if imgL.size[1] % 32 != 0:
                            times = imgL.size[1]//32       
                            top_pad = (times+1)*32 -imgL.size[1]
                        else:
                            top_pad = 0
                        if imgL.size[0] % 32 != 0:
                            times = imgL.size[0]//32                    
                            right_pad = (times+1)*32-imgL.size[0]
                        else:
                            right_pad = 0    
                        if top_pad !=0 and right_pad != 0:
                            est_disp = est_disp[top_pad:,:-right_pad]
                        elif top_pad ==0 and right_pad != 0:
                            est_disp = est_disp[:,:-right_pad]
                        elif top_pad !=0 and right_pad == 0:
                            est_disp = est_disp[top_pad:,:]
                        else:
                            est_disp = est_disp

                        width, _ = imgL.size
                        print(est_disp.shape)

                       #+predict_save
                        focal=  width / (2.0 * np.tan((fov* np.pi) / 360.0))
                        print(focal)
                        c=all_metrics(ref_disp=ref_disp, depth=None, est_disp=est_disp, focal=focal, baseline=baseline, ranges=[0.5,50], str_label=left_filenames[index].split('/')[-4]+left_filenames[index].split('/')[-3]+left_filenames[index].split('/')[-1])#(ref_disp, depthimg, est_disp, focal, baseline, [0.5 ,50], name)
                        #fig, hist_fig,c=plot_stuff(imgL, imgR, ref_disp, depthimg_norm, est_disp, focal, baseline, ranges=[0.5, 50], str_label=left_filenames[index].split('/')[-4]+left_filenames[index].split('/')[-3]+left_filenames[index].split('/')[-1])
                        if(os.path.isdir(ErrorsDirectory+"/")==False):
                            os.mkdir(ErrorsDirectory+"/")
                        #if(os.path.isdir(ErrorsDirectory+"/"+p.split("/")[1])==False):
                        #    os.mkdir(ErrorsDirectory+"/"+p.split("/")[1])
                        """fig_name=left_filenames[index].split('/')[-4]+left_filenames[index].split('/')[-3]+left_filenames[index].split('/')[-1]
                        if(c["mdre"]>10):
                            if(os.path.isdir(ErrorsDirectory+"/"+"/10_inf")==False):
                                os.mkdir(ErrorsDirectory+"/"+"/10_inf")
                            fig.savefig(ErrorsDirectory+"/"+"/10_inf/"+fig_name)
                        if(c["mdre"]>4 and c["mdre"]<=10):
                            if(os.path.isdir(ErrorsDirectory+"/"+"/4_10")==False):
                                os.mkdir(ErrorsDirectory+"/"+"/4_10")
                            fig.savefig(ErrorsDirectory+"/"+"/4_10/"+fig_name)
                        if(c["mdre"]>2 and c["mdre"]<=4):
                            if(os.path.isdir(ErrorsDirectory+"/"+"/2_4")==False):
                                os.mkdir(ErrorsDirectory+"/"+"/2_4")
                            fig.savefig(ErrorsDirectory+"/"+"/2_4/"+fig_name)
                        if(c["mdre"]<=2):
                            if(os.path.isdir(ErrorsDirectory+"/"+"/0_2")==False):
                                os.mkdir(ErrorsDirectory+"/"+"/0_2")
                            fig.savefig(ErrorsDirectory+"/"+"/0_2/"+fig_name)"""
##
                        #hist_fig.savefig(ErrorsDirectory+"/hist_"+name)
                        #hist_fig.clf()
                        #fig.clf()
                        plt.clf()
                        gc.collect()
                        plt.close("all")
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/D1","a")
                        #f.write(path+p+dep+name+" "+str(c["d1"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/mdre","a")
                        #f.write(path+p+dep+"/"+name+" "+str(c["mdre"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/mre","a")
                        #f.write(path+p+dep+name+" "+str(c["mre"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/mse","a")
                        #f.write(path+p+dep+name+" "+str(c["mse"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/sze","a")
                        #f.write(path+p+dep+name+" "+str(c["sze"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/bmp","a")
                        #f.write(path+p+dep+name+" "+str(c["bmp"])+"\n")
                        #f.close()
                        #f=open(ErrorsDirectory+"/"+p.split("/")[1]+"/bmpre","a")
                        #f.write(path+p+dep+name+" "+str(c["bmpre"])+"\n")
                        #f.close()

                        f=open(ErrorsDirectory+"/D1","a")
                        f.write(left_filenames[index]+" "+str(c["d1"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/mdre","a")
                        f.write(left_filenames[index]+" "+str(c["mdre"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/mre","a")
                        f.write(left_filenames[index]+" "+str(c["mre"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/mse","a")
                        f.write(left_filenames[index]+" "+str(c["mse"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/sze","a")
                        f.write(left_filenames[index]+" "+str(c["sze"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/bmp","a")
                        f.write(left_filenames[index]+" "+str(c["bmp"])+"\n")
                        f.close()
                        f=open(ErrorsDirectory+"/bmpre","a")
                        f.write(left_filenames[index]+" "+str(c["bmpre"])+"\n")
                        f.close()
                        print(str(c["mdre"]))
        mdre=sum_mean(ErrorsDirectory, ErrorsDirectory+"/Error")
        sum_mean_excel(ErrorsDirectory,ErrorsDirectory +"/EXCEL")
        #shutil.rmtree(ErrorsDirectory)
        return mdre
if __name__ == '__main__':
    Error_metrcs_save="/media/developer/Vidar/Fast-ACVNet-Daniel/Errors_metrics/config1"
    list_filenames='/media/developer/Vidar/Fast-ACVNet-Daniel/fisheye/fisheye_test_1'
    save_dir = '/media/developer/Vidar/Fast-ACVNet-Daniel/config1_predict'
    #bs='./baseline_september'
    data_path='/media/developer/Vidar/Dataset_poprawione/'
    m=errors_methric(Error_metrcs_save,list_filenames, save_dir,73.46566761132043,data_path )