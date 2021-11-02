from pcdet.models import backbones_2d
from torch_scatter import scatter_mean
from torch.nn import functional as F
from torch import nn
import numpy as np
import torch 

def voxelization(points, pc_range, voxel_size):    
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) * (points[:, 2] <= pc_range[5])
    points = points[keep, :]    
    coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    voxels = scatter_mean(points, inverse_indices, dim=0)
    return voxels, unique_coords


class DynamicVoxelEncoder(nn.Module):
    def __init__(
        self, model_cfg, num_point_features,point_cloud_range,voxel_size,grid_size,**kwargs
    ):
        super(DynamicVoxelEncoder, self).__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features
        self.pc_range = torch.tensor(point_cloud_range) 
        self.voxel_size = torch.tensor(voxel_size) 

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        
        points = batch_dict['points']

        voxels = []
        coors = []

        for i in range(batch_dict['batch_size']):
            point = points[points[:,0]==i][:,1:]
            voxel,coor = voxelization(point,self.pc_range.to(point.device),self.voxel_size.to(point.device))
            voxels.append(voxel)
            coors.append(coor)

        coors_batch = []

        for i in range(len(voxels)):
            coor_pad = F.pad(coors[i], (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        voxels_batch = torch.cat(voxels, dim=0)

        batch_dict['voxel_features'] = voxels_batch
        batch_dict['voxel_coords'] = coors_batch

        return batch_dict