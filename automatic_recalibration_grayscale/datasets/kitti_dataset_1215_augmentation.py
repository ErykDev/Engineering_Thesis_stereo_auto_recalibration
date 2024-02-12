import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
# import torchvision.transforms as transforms
from . import flow_transforms
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class KITTIDataset_ag(Dataset):
    def __init__(self, kitti15_datapath, list_filename, training):
        self.datapath_15 = kitti15_datapath
        #self.datapath_12 = kitti12_datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        im = Image.open(filename).convert('RGB')
        im = im.resize((288, 256), Image.LANCZOS)
        return im

    def load_disp(self, filename):
        data = Image.open(filename)
        data = data.resize((288,256), Image.NEAREST)
        data = np.array(data, dtype=np.float32) / 256.
        
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        # left_name = self.left_filenames[index].split('/')[1]
        # if left_name.startswith('image'):
        self.datapath = self.datapath_15
        # else:
            # self.datapath = self.datapath_12
        
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None
        # random crop
        w, h = left_img.size
        crop_w, crop_h = 288, 256  # simil
        # x1 = random.randint(0, w - crop_w)
        # y1 = random.randint(0, h - crop_h)
        # left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        # right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        # disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
        if self.training:
            th, tw = 256, 288
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)
            
            random_brightness_solo = np.random.uniform(0.5, 1.0, 2)
            random_gamma_solo = np.random.uniform(0.95, 1.05, 2)
            random_contrast_solo = np.random.uniform(0.95, 1.05, 2)
            random_saturation_solo = np.random.uniform(0, 0.4, 2)
            if np.random.rand() > 0.5:
                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
                left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])
                right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])
            elif np.random.rand() > 0.5:
                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness_solo[0])              
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma_solo[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast_solo[0])
                left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation_solo[0])
            elif np.random.rand() > 0.5:
                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness_solo[1])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma_solo[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast_solo[1])
                right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation_solo[1])
            # right_img = np.array(right_img)
            # left_img = np.array(left_img)

            # # geometric unsymmetric-augmentation
            # angle = 0
            # px = 0
            # if np.random.binomial(1, 0.5):
                # # angle = 0.1;
                # # px = 2
                # angle = 0.05
                # px = 1
            # co_transform = flow_transforms.Compose([
                # # flow_transforms.RandomVdisp(angle, px),
                # # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                # flow_transforms.RandomCrop((th, tw)),
            # ])
            # augmented, disparity = co_transform([left_img, right_img], disparity)
            # left_img = augmented[0]
            # right_img = augmented[1]

            # right_img.flags.writeable = True
            # if np.random.binomial(1,0.2):
            #   sx = int(np.random.uniform(35,100))
            #   sy = int(np.random.uniform(25,75))
            #   cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
            #   cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
            #   right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            # random patch exchange of right image
            patch_h = random.randint(35, 90)
            patch_w = random.randint(25, 75)
            patch1_x = random.randint(0, crop_h-patch_h)
            patch1_y = random.randint(0, crop_w-patch_w)
            patch2_x = random.randint(0, crop_h-patch_h)
            patch2_y = random.randint(0, crop_w-patch_w)
            # pdb.set_trace()
            # print(right_img.shape)
            img_patch = right_img[:, patch2_x:patch2_x+patch_h, patch2_y:patch2_y+patch_w]
            right_img[:, patch1_x:patch1_x+patch_h, patch1_y:patch1_y+patch_w] = img_patch
            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)

            disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)




            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low": disparity_low}
        else:
            w, h = left_img.size

            # normalize
            
            
            

            # pad to size 1248x384
            if h % 32 != 0:
                times = h//32       
                top_pad = (times+1)*32 -h
            else:
                top_pad = 0
            if w % 32 != 0:
                times = w//32                    
                right_pad = (times+1)*32-w
            else:
                right_pad = 0    
            

            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
