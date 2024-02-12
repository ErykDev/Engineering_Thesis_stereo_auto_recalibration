from torch.autograd import Variable
from torch import Tensor
import torch

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                # print("masks[idx].float().mean() too small, skip")
                pass
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()
        
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    
    return torch.mean(err_mask.float())*100

@make_nograd_func
@compute_metric_for_each_image
def mdre_metric(D_est, D_gt, mask, focal, baseline):
    
    index = D_est > 0.0
    gt_index = D_gt > 0.0
    depth_est = torch.zeros_like(D_est)
    depth_gt = torch.zeros_like(D_gt)
    depth_est[index] = focal * baseline / D_est[index]
    depth_gt[gt_index] = focal * baseline / D_gt[gt_index]

    depth_est, depth_gt = depth_est[mask], depth_gt[mask]

    depth_err = (torch.abs(depth_gt - depth_est) / depth_gt) * 100
    
    idxs = depth_err > 100
    depth_err[idxs] = 100
    
    return torch.mean(depth_err) 
