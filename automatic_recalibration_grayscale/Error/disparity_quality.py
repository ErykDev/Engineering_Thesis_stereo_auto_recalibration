import numpy as np
import math
from matplotlib import pyplot as plt
kernel=3
step=3
def second_smallest(data):
    data = data.flatten()
    data = data.tolist()
    data.sort()
 
    for num in data:
        if num > 0.0:
            return num
    return 0.0
 
def dilation_step(data, dilation_level=2):
    image_src = data
    
    orig_shape = image_src.shape
    pad_width = dilation_level - 2
 
    # pad the image with pad_width
    image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
 
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
    
    # obtain the submatrices according to the size of the kernel
    flat_submatrices = np.array([
        image_pad[i:(i + dilation_level), j:(j + dilation_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])
    
    # replace the values either 255 or 0 by dilation condition
    #image_dilate = np.array([np.partition(i, 1)[0] if (np.partition(i, 1)[0] != 0).all() else np.partition(i, 1)[1] for i in flat_submatrices])
    image_dilate = np.array([second_smallest(i) for i in flat_submatrices])
 
    # obtain new matrix whose shape is equal to the original image size
    return image_dilate.reshape(orig_shape)
 
def dilate(src, steps, kernel):
    data = src
    
    for i in range(steps):
        data = dilation_step(data, dilation_level=kernel)
    
    return data
class disparityQuality():

    def __init__(self, ref_disp, disp, focal=None, baseline=None, ref_depth=None, depth_range=None):
        
        self.focal = focal
        self.baseline = baseline
        self.ref_disp = ref_disp
        self.disp = disp

        if depth_range is not None:
            self.depth_range = depth_range
        
        else:
            self.depth_range = [0.0, math.inf]

        if ref_depth is not None:
            #print(ref_depth.size)
            self.depth_norm = ref_depth
            self.ref_depth = self.destDepth()
            
            
        
        else:
            
            self.ref_depth = self.baseline * self.focal / self.ref_disp

        if ref_disp is None:
            ref_disp=np.zeros(self.depth_norm.shape)
            self.ref_disp=self.baseline * self.focal / self.depth_norm
            idx = self.ref_disp < 0.5
            self.ref_disp[idx] = 0
            idx=self.ref_disp>191
            self.ref_disp[idx] = 0
            
        self.disp_bin = self.binarize(self.ref_disp)
        
    def imgDepthNormalization(self, depth):
        """ Method for normalize depth in RGB to disparity """

        array = np.array(depth)
        array = array.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        #normalized_depth = np.dot(array[:, :, :], [1.0, 256.0, 65536.0])
        #normalized_depth = normalized_depth / ((256.0 * 256.0 * 256.0)-1)  # (256.0 * 256.0 * 256.0 - 1.0)
        normalized_depth=np.zeros(array.shape)
        normalized_depth = array/256.0
        
        return normalized_depth
    
    def binarize(self, disp):
        """ Method for binarize reference disparity """
        disp = disp.astype(np.float32)
        disp_bin = np.where(disp > 0.0, 255.0, 0.0)

        return disp_bin

    def mseError(self):
        """ Calculate MEAN SQUARE ERROR """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp 

        idx = s_all == 255
        d_gt = d_gt[idx].astype(np.float32)
        d_est = d_est[idx].astype(np.float32)
        sub_mse = np.zeros_like(d_gt)
        idx = d_gt != 0
        sub_mse[idx] = (d_gt[idx] - d_est[idx]) / 4
        sub_mse_pow = sub_mse[idx] * sub_mse[idx]

        return np.mean(sub_mse_pow)

    def mreError(self):
        """ Calculate MEAN RELATIVE ERROR """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp   
        idx = s_all == 255
        d_gt = d_gt[idx].astype(np.float32)
        d_est = d_est[idx].astype(np.float32)
        sub_mre = np.zeros_like(d_gt)
        idx = d_gt != 0
        sub_mre[idx] = np.abs((d_gt[idx] - d_est[idx]) / d_gt[idx])
        return np.mean(sub_mre)
    
    def szeError(self):
        """ Calculate Sigma-Z Error """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp 

        idx = s_all == 255
        mi = 1
        d_gt = d_gt[idx].astype(np.float32)
        d_est = d_est[idx].astype(np.float32)
        sub_sze = np.zeros_like(d_gt)
        idx = d_gt != 0
        sub_sze[idx] = np.abs((4 / (d_gt[idx] + mi)) - (4 / (d_est[idx] + mi)))

        return np.sum(sub_sze)

    def bmpError(self):
        """ Calculate Bad Matching Pixel Error """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp 

        ball = []
        idx = s_all == 255
        idx_nnz = np.count_nonzero(idx)
        d_gt = d_gt.astype(np.float32)
        d_est = d_est.astype(np.float32)

        sub_bmp = np.abs(d_est - d_gt) > 4
        b = np.logical_and(sub_bmp, s_all == 255)
        ball  = np.count_nonzero(b)   

        return (ball / idx_nnz) * 100
    
    def bmpreError(self):
        """ Calculate Bad Matching Pixel Relative Error """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp 

        d_gt = d_gt.astype(np.float32)
        d_est = d_est.astype(np.float32)
        delta = np.abs(d_gt - d_est)
        rho = np.zeros_like(d_gt)
        idx_sgt = d_gt != 0
        rho[idx_sgt] = delta[idx_sgt] / d_gt[idx_sgt]
        B = delta > 4
        Ball = np.logical_and(B, s_all == 255)
        idx = Ball == 1
        bmpre_all = np.sum(rho[idx])

        return bmpre_all
    
    def depthMask(self):
        """ Method for return calucalte mask image """
        if self.ref_depth is not None:
            d_gt = np.array(self.ref_depth)
            ranges = self.depth_range
            d_gt_mask = np.where((d_gt > ranges[0]) & (d_gt < ranges[1]), 255.0, 0.0)

            return d_gt_mask

    def depthMetric(self):

        """ VIDAR proposal depth reconstruction error """
        if self.ref_depth is not None:

            d_gt = np.array(self.ref_depth)
            d_est = self.focal * self.baseline / self.disp
            ranges = self.depth_range
            
            idx = (d_gt < ranges[1]) & (d_gt > ranges[0]) & (d_est < ranges[1])
            d_gt = d_gt[idx]
            d_est = d_est[idx]

            depth_err = (np.abs(d_gt - d_est) / d_gt) * 100
            
            idxs = depth_err > 100
            depth_err[idxs] = 100

            depth_err = np.mean(depth_err.astype(np.float32)) 
            
            return depth_err
    
    def destDepth(self):
        depth = self.depth_norm
        ranges = self.depth_range
        idx = (depth > ranges[1]) | (depth < ranges[0])
        depth[idx] = math.inf

        return depth
    
    def metricD1(self):
        """ Calculate D1 metric """
        d_gt = self.ref_disp
        s_all = self.disp_bin
        d_est = self.disp
        idx = s_all == 255
        d_gt = d_gt.astype(np.float32)
        d_est = d_est.astype(np.float32)
        d_est, d_gt = d_est[idx], d_gt[idx]
        E = np.abs(d_gt - d_est)
        err_mask = (E > 3) & (E / np.abs(d_gt) > 0.05)

        return np.mean(err_mask.astype(np.float32)) * 100
    
    def __dispErrorMap(self, d_gt, d_est):
        
        E = np.zeros_like(d_gt)
        idx = d_gt > 0
        E[idx] = np.abs(d_gt[idx] - d_est[idx])
        # E = np.abs(d_gt - d_est)
        return E
    
    def __errorColorMap(self):
        cols = np.array([ [0/3.0,       0.1875/3.0,  49,  54, 149],
         [0.1875/3.0,  0.375/3.0,   69, 117, 180],
         [0.375/3.0,   0.75/3.0,   116, 173, 209],
         [0.75/3.0,    1.5/3.0,    171, 217, 233],
         [1.5/3.0,     3/3.0,      224, 243, 248],
         [3/3.0,       6/3.0,      254, 224, 144],
         [6/3.0,      12/3.0,      253, 174,  97],
        [12/3.0,      24/3.0,      244, 109,  67],
        [24/3.0,      48/3.0,      215,  48,  39],
        [48/3.0,     math.inf,      165,   0,  38 ]])
        cols[:,2:5] = cols[:,2:5] / 255

        return cols

    def dispErrorToImage(self):

        d_gt = self.ref_disp
        d_est = self.disp
        s_all = self.disp_bin
        idx = s_all == 255
        tau = [3, 0.05]
        d_gt = d_gt.astype(np.float32)
        d_est = d_est.astype(np.float32)
        E = self.__dispErrorMap(d_gt, d_est)
        cols = self.__errorColorMap()
        E1 = np.zeros_like(d_gt)
        E2 = np.zeros_like(d_gt)
        E1[idx] = E[idx]/tau[0]
        E2[idx] = (E[idx]/np.abs(d_gt[idx]))/tau[1]
        E = np.minimum(E1, E2)
        n, m = d_gt.shape
        d_err = np.zeros((n,m,3))
        hist_ranges = []
        sum_occ = len(E[idx])

        for i in range(0,10):
            v, u = np.where((E >= cols[i, 0]) & (E <= cols[i, 1]) & (d_gt > 0))
            idx = (E >= cols[i, 0]) & (E <= cols[i, 1]) & (d_gt > 0)          
            idxs = np.where(idx == True)
            hist_ranges.append(self.__computePercent(sum_occ, len(idxs[1])))
            d_err[v,u,0] = cols[i, 2]
            d_err[v,u,1] = cols[i, 3]
            d_err[v,u,2] = cols[i, 4]

        return d_err, hist_ranges

    def __depthErrorMap(self, depth_gt, depth_est, ranges):
        
        h, w = depth_gt.shape
        E = np.zeros((h, w))

        idx = (depth_gt < ranges[1]) & (depth_gt > ranges[0]) & (depth_est < ranges[1])
        E[idx] = (np.abs(depth_gt[idx] - depth_est[idx]) / depth_gt[idx]) * 100
        idxs = E > 100
        E[idxs] = 100

        return E
    
    def __absoluteErrorDepth(self, depth_gt, depth_est, ranges, depth_all):
        
        cols = np.array([ [0.0, 0.5], [0.5, 0.8],
                        [0.8, 1.2], [1.2, 1.6],
                        [1.6, 2.0], [2.0, 4.0],
                        [4.0, 8.0], [8.0, 12.0],
                        [12.0, 18.0], [18.0, math.inf] ])

        h, w = depth_gt.shape
        A = np.zeros((h, w))

        idx = (depth_gt < ranges[1]) & (depth_gt > ranges[0]) & (depth_est < ranges[1])
        A[idx] = np.abs(depth_gt[idx] - depth_est[idx])
        sum_occ = len(A[idx])
        hist_ranges = []

        for i in range(0, 10):
            idx = (A >= cols[i, 0]) & (A <= cols[i, 1]) & (depth_all == 255) 
            idxs = np.where(idx == True)
            hist_ranges.append(self.__computePercent(sum_occ, len(idxs[1])))

        return hist_ranges

    def __errorDepthColorMap(self):
        cols = np.array([[0.0,       0.7,  49,  54, 149],
        [0.7,    1.4,     69, 117, 180],
        [1.4,    2.0,     116, 173, 209],
        [2.0,   4.0,     171, 217, 233],
        [4.0,   8.0,     224, 243, 248],
        [8.0,   12.0,     254, 224, 144],
        [12.0,   15.0,     253, 174,  97],
        [15.0,   17.5,     244, 109,  67],
        [17.5,   20.0,     215,  48,  39],
        [20.0,   math.inf, 165,   0,  38 ]])
        cols[:,2:5] = cols[:,2:5] / 255

        return cols
    
    def depthErrorToImage(self):

        if self.ref_depth is not None:
            depth_gt = self.ref_depth
            depth_est = self.focal * self.baseline / self.disp
            depth_all = self.depthMask()
            ranges = self.depth_range
            E = self.__depthErrorMap(depth_gt, depth_est, ranges)
            #E=dilate(E,4,4)
            cols = self.__errorDepthColorMap()
            idx = depth_all == 255
            n, m = depth_gt.shape
            d_err = np.zeros((n,m,3))
            hist_ranges = []
            index = (depth_gt > ranges[0]) & (depth_gt < ranges[1])
            sum_occ = len(E[index])
            
            for i in range(0,10):
                v, u = np.where((E >= cols[i, 0]) & (E <= cols[i, 1]) & (depth_gt < ranges[1]))            
                idx = (E >= cols[i, 0]) & (E <= cols[i, 1]) & (depth_all == 255) 
                idxs = np.where(idx == True)
                hist_ranges.append(self.__computePercent(sum_occ, len(idxs[1])))
                d_err[v,u,0] = cols[i, 2]
                d_err[v,u,1] = cols[i, 3]
                d_err[v,u,2] = cols[i, 4]
                if(v+1<n):
                    d_err[v+1,u,0] = cols[i, 2]
                    d_err[v+1,u,1] = cols[i, 3]
                    d_err[v+1,u,2] = cols[i, 4]
                if(u+1<m):
                    d_err[v,u+1,0] = cols[i, 2]
                    d_err[v,u+1,1] = cols[i, 3]
                    d_err[v,u+1,2] = cols[i, 4]
                
                    
                
                
                
                
            

            abs_hist_ranges = self.__absoluteErrorDepth(depth_gt, depth_est, ranges, depth_all)

        return d_err, hist_ranges, abs_hist_ranges

    def __computePercent(self, sum, x):
        if(sum!=0):
            percent = 100 * x / sum
        else:
            percent ="---"

        return percent



        
        
        
        
        
