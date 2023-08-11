import torch
import numpy as np

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).cuda()   # [B, npoint]
    distance = torch.ones(B, N).cuda() * 1e10                     # [B,N] 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).cuda() # [B,] random init first center's index
    batch_indices = torch.arange(B, dtype=torch.long).cuda()      # [B,] ~ [0,1,бнбн,B-1]
    for i in range(npoint):
        centroids[:, i] = farthest                                # i=0 centroids[:,0] = init, i>0, farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 
        dist = torch.sum((xyz - centroid) ** 2, -1)               # all points' distance to centroid
        mask = dist < distance                                    # mask the dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
    
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
    
class FPSMix(object):
    def __init__(self):
        pass

    def __call__(self, pc, target, mg):
        bsize, Npoint, dim = pc.size()
        rand_index = torch.randperm(bsize).cuda()
        
        if np.random.uniform() < mg:
            target_b = target[rand_index]
            pc = torch.cat((pc, pc[rand_index]), dim=1)
            
            idx = farthest_point_sample(pc, Npoint)
            
            pc_idx_min = torch.min(idx, dim=1).values.reshape(bsize,1).repeat(1,Npoint)
            pc_idx_max = torch.max(idx, dim=1).values.reshape(bsize,1).repeat(1,Npoint)
            idx1 = torch.where(idx < Npoint, idx, pc_idx_min)
            idx2 = torch.where(idx >= Npoint, idx, pc_idx_max)
            pc1 = index_points(pc, idx1)
            pc2 = index_points(pc, idx2)
            
            pc = index_points(pc, idx)
            lam = idx < Npoint
            
            lam = torch.sum(lam, 1).float() / Npoint

        else:
            target_b = target.detach()
            lam = torch.ones(bsize).cuda()
            
            pc1 = pc
            pc2 = pc
            
        return pc, target, target_b, lam , pc1, pc2  