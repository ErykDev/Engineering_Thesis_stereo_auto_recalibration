from PIL.Image import merge
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from disparity_quality import disparityQuality as ds
import math
from matplotlib import cm
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

COLORS = [(49/256,   54/256, 149/256, 1),
          (69/256,  117/256, 180/256, 1),
          (116/256, 173/256, 209/256, 1),
          (171/256, 217/256, 233/256, 1),
          (224/256, 243/256, 248/256, 1),
          (254/256, 224/256, 144/256, 1),
          (253/256, 174/256,  97/256, 1),
          (244/256, 109/256,  67/256, 1),
          (215/256,  48/256,  39/256, 1),
          (165/256,   0/256,  38/256, 1)]

RANGES_DISP = [0.0,  0.19, 0.38,
		  0.75, 1.5, 3,
		  6,  12, 24,
		  48, 'inf']

RANGES_DEPTH = [0,    0.7,   1.4,
		        2,    4, 8,
		        12, 15,  17.5,
		        20,   'inf']
		
TICKS_DISP = [0.0, 0.1, 0.2, 0.3, 0.4,
		 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TICKS_DEPTH = [0.0, 0.1, 0.2, 0.3, 0.4,
		 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

cmap_name = ['disp', 'depth']

norm = mpl.colors.Normalize(vmin = 0, vmax = 192)

def create_cmap(name):
	
	cmap = LinearSegmentedColormap.from_list(name, COLORS, N=10)

	return cmap

def all_metrics(ref_disp, depth, est_disp, focal, baseline, ranges, str_label):
	

	quality =  ds(ref_disp, est_disp, focal=focal, baseline=baseline, ref_depth=depth, depth_range=ranges)

	d1 = quality.metricD1()
	mdre = quality.depthMetric()
	mre = quality.mreError()
	mse = quality.mseError()
	sze = quality.szeError()
	bmp = quality.bmpError()
	bmpre = quality.bmpreError()
	
	metrics={"d1": d1, "mdre": mdre, "mre": mre,
			 "mse": mse, "sze": sze, "bmp": bmp, "bmpre": bmpre}

	return metrics

def plot_stuff(imgL, imgR, ref_disp, depth, est_disp, focal, baseline, ranges, str_label):
	norm_depth = mpl.colors.Normalize(vmin = ranges[0], vmax = 150)
	quality = ds(ref_disp, est_disp, focal=focal, baseline=baseline, ref_depth=depth, depth_range=ranges)
	est_depth = np.zeros_like(est_disp)
	est_idx = est_disp > 0
	est_depth = (focal * baseline) / est_disp
	

	d1 = quality.metricD1()
	mdre = quality.depthMetric()
	mre = quality.mreError()
	mse = quality.mseError()
	sze = quality.szeError()
	bmp = quality.bmpError()
	bmpre = quality.bmpreError()
	ref_disp=quality.ref_disp
	ref_disp=dilate(ref_disp,step,kernel)

	metrics = {"d1": d1,
				"mdre": mdre,
				"mre": mre,
				"mse": mse,
				"sze": sze,
				"bmp": bmp,
				"bmpre": bmpre}

	dispErrorMap, disp_hist_ranges = quality.dispErrorToImage()
	depthErrorMap, depth_hist_ranges, abs_hist_ranges = quality.depthErrorToImage()
	est_disp=dilate(est_disp,step,kernel)
	est_depth=dilate(est_depth,step,kernel)
	depth=dilate(depth,step,kernel)

	fig = plt.figure(figsize=(10, 12), dpi=150)
	metric_txt = "MRE: {:.3f} MSE: {:.2f} SZE: {:.0f} BMP: {:.3f} BMPRE: {:.0f}".format(mre, mse, sze, bmp, bmpre)
	fig.suptitle("{}\n{}".format(str_label, metric_txt), fontsize = 'x-large')

	fig.add_subplot(4, 2, 1)
	plt.imshow(imgL)
	plt.axis('off')
	plt.title("LEFT IMAGE")
	
	fig.add_subplot(4, 2, 2)
	plt.imshow(imgR)
	plt.axis('off')
	plt.title("RIGHT IMAGE")

	fig.add_subplot(4, 2, 3)
	plt.imshow(ref_disp, cmap='GnBu_r', norm=norm)
	plt.axis('off')
	plt.title("GROUND TRUTH DISPARITY")

	fig.add_subplot(4, 2, 4)
	plt.imshow(est_disp, cmap='GnBu_r', norm=norm)
	plt.axis('off')
	plt.title("ESTIMATED DISPARITY")

	fig.add_subplot(4, 2, 5)
	plt.imshow(depth, norm=norm_depth)
	plt.axis('off')
	plt.title("GROUND TRUTH DEPTH")
	plt.colorbar(orientation='horizontal', norm=norm_depth)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.tight_layout()

	fig.add_subplot(4, 2, 6)
	plt.imshow(est_depth, norm=norm_depth)
	plt.axis('off')
	plt.title("ESTIMATED DEPTH")
	plt.colorbar(orientation='horizontal')
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.tight_layout()

	fig.add_subplot(4, 2, 7)
	plt.imshow(dispErrorMap, cmap= create_cmap(cmap_name[0]))
	plt.axis('off')
	plt.title("D1: {:.2f} [%]".format(d1))
	cbar_disp = plt.colorbar(orientation='horizontal', ticks = TICKS_DISP, shrink = 0.75)
	cbar_disp.set_ticklabels(RANGES_DISP)

	# fig.add_subplot(4, 2, 6)
	# metric_txt = "MRE: {:.3f} \nMSE: {:.2f}\nSZE: {:.0f}\nBMP: {:.3f}\nBMPRE: {:.0f}".format(mre, mse, sze, bmp, bmpre)
	# plt.text(0, 0, metric_txt, style='italic', fontsize = 'x-large',
    #     bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, position=(0.3, 0.3))
	# plt.axis('off')

	fig.add_subplot(4, 2, 8)
	plt.imshow(depthErrorMap, cmap= create_cmap(cmap_name[1]))
	plt.axis('off')
	plt.title("MDRE: {:.2f} [%]".format(mdre))
	cbar_depth = plt.colorbar(orientation='horizontal', ticks = TICKS_DEPTH, shrink = 0.75)
	cbar_depth.set_ticklabels(RANGES_DEPTH)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)


	hist_fig = hists_plot(disp_hist_ranges, depth_hist_ranges, abs_hist_ranges, str_label)
	del quality

	return fig, hist_fig, metrics

def hists_plot(disp_hist, depth_hist, abs_hist_ranges, str_label):

	disp_hist_len = len(disp_hist)
	depth_hist_len = len(depth_hist)

	x_disp = np.linspace(0, disp_hist_len, disp_hist_len)
	x_depth = np.linspace(0, depth_hist_len, depth_hist_len)
	x_ticks_disp = ['0 - 0.19', '0.19 - 0.38', '0.38 - 0.75',
			   '0.75 - 1.5', '1.5 - 3', '3 - 6',
			   '6 - 12', '12 - 24', '24 - 48', '48 - inf']

	x_ticks_depth = ['0 - 0.7', '0.7 - 1.4', '1.4 - 2',
			   '2 - 4', '4 - 8', '8 - 12',
			   '12 - 15', '15 - 17.5', '17.5 - 20', '20 - inf']
	
	x_ticks_abs = ['0 - 0.5', '0.5 - 0.8', '0.8 - 1.2',
			   '1.2 - 1.6', '1.6 - 2', '2 - 4',
			   '4 - 8', '8 - 12', '12 - 18', '18 - inf']

	fig = plt.figure(figsize=(7,7))
	fig.suptitle(str_label, fontsize = 'large')
	bar_width = 0.35
	
	fig.add_subplot(3, 1, 1)
	bar = plt.bar(x_disp, disp_hist, bar_width)
	plt.title("Disp error map histogram")
	plt.ylim((0, 100))
	plt.xlim((-1, 11))
	plt.xticks(x_disp, x_ticks_disp, rotation = 25)
	plt.ylabel("[%]")
	for rect in bar:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width(), height, "{:.1f}".format(height), ha="center", va="bottom")
	
	fig.add_subplot(3, 1, 2)
	bar = plt.bar(x_depth, depth_hist, bar_width, align='center')
	plt.title("Depth error map histogram")
	plt.ylim((0, 100))
	plt.xlim((-1, 11))
	plt.xticks(x_depth, x_ticks_depth, rotation = 25)
	plt.ylabel("[%]")
	plt.xlabel("[%]")
	for rect in bar:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width(), height, "{:.1f}".format(height), ha="center", va="bottom")

	fig.add_subplot(3, 1, 3)
	bar = plt.bar(x_depth, abs_hist_ranges, bar_width, align='center')
	plt.title("Absolute depth map error histogram")
	plt.ylim((0, 100))
	plt.xlim((-1, 11))
	plt.xticks(x_depth, x_ticks_abs, rotation = 25)
	plt.ylabel("[%]")
	plt.xlabel("[m]")
	for rect in bar:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width(), height, "{:.1f}".format(height), ha="center", va="bottom")
	plt.tight_layout()

	return fig

def train_plotter(d1_list, loss_list, mdre_list, name):

	fig = plt.figure(figsize=(7,10), dpi=150)
	fig.suptitle('{} train metrics'.format(str(name)))

	fig.add_subplot(3, 1, 1)
	plt.plot(loss_list)
	plt.ylabel("loss")
	plt.xlabel("epoch")

	fig.add_subplot(3, 1, 2)
	plt.plot(d1_list)
	plt.ylabel("D1")
	plt.xlabel("epoch")

	fig.add_subplot(3, 1, 3)
	plt.plot(mdre_list)
	plt.ylabel("MDRE")
	plt.xlabel("epoch")

	return fig
