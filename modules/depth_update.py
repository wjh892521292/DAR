import torch
import torch.nn.functional as F
import copy



def update_sample(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range):
    
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        mode = '2'
        if mode == 'direct':
            depth_range = uncertainty_range
            depth_start_update = torch.clamp_min(depth_r - 0.5 * depth_range, min_depth)
        elif mode == '2':
            depth_range = uncertainty_range + (target_bin_right - target_bin_left).abs()
            depth_start_update = torch.clamp_min(target_bin_left - 0.5 * uncertainty_range, min_depth)
            
        interval = depth_range / depth_num
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
    return bin_edges.detach(), curr_depth.detach()


def update_sample3(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range):
    
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        mode = 'direct'
        if mode == 'direct':
            depth_range = uncertainty_range
            depth_r = upsample(depth_r, scale_factor=2)
            depth_range = upsample(depth_range, scale_factor=2)
            depth_start_update = torch.clamp_min(depth_r - 0.5 * depth_range, min_depth)
        else:
            depth_range = uncertainty_range + (target_bin_right - target_bin_left).abs()
            depth_start_update = torch.clamp_min(target_bin_left - 0.5 * uncertainty_range, min_depth)

        interval = depth_range / depth_num
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones([b, 1, h*2, w*2], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
    return bin_edges.detach(), curr_depth.detach()

def update_sample2(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range, scale):
    
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

     
      

        # for i in range(16):
        #     depth_idx = torch.clamp(target_bin_left * scale - (16-scale)//2 + i, 0, 127).long()
        #     print(depth_idx.shape)
        #     mask = torch.scatter(mask, dim=1, index=depth_idx, src=mask_1.float())

        ### depth_idx: 下一个预测时候有效区域下标： b, 16, h, w
        depth_idx = torch.cat([torch.clamp(target_bin_left * scale - (16-scale)//2 + i, 0, 127).long() for i in range(16)], dim=1)
        # print(depth_idx.shape)

        depth_range = max_depth - min_depth
        depth_num = depth_num * scale
        interval = depth_range / depth_num
        interval = interval * torch.ones([b, 1, h, w], device=bin_edges.device)
        interval = interval.repeat(1, depth_num, 1, 1)
        interval =torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        ###下一次预测的分组 b, depth_num, h, w
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) 
        # print(curr_depth.shape, depth_num)

        ###下一次预测有效分组bin值作为input encoder b, 16, h, w
        ava_curr_depth = torch.gather(curr_depth, dim=1, index=depth_idx) 
        # print(ava_curr_depth.shape, depth_num)

        
        mask = torch.ones((b, 128, h, w), device=bin_edges.device) * 0.2
        mask_1 = torch.ones((b, 128, h, w), device=bin_edges.device)
        ###下一次预测的mask，有效区域为1，其他区域为0.2
        mask = torch.scatter(mask, dim=1, index=depth_idx, src=mask_1.float())
        # print(mask.shape, depth_num)

    return depth_num, bin_edges.detach(), curr_depth.detach(), ava_curr_depth.detach(), mask.detach()


def get_label(gt_depth_img, bin_edges, depth_num):

    with torch.no_grad():
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device)
        for i in range(depth_num):
            bin_mask = torch.ge(gt_depth_img, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, 
                torch.lt(gt_depth_img, bin_edges[:, i + 1]))
            gt_label[bin_mask] = i
        
        return gt_label

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
