import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import open3d as o3d
from numpy.linalg import norm


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
                                            #npoint, radius_list, nsample_list, in_channel, mlp_list
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #torch.Size([1, 128, 1024])
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) #torch.Size([1, 128, 2048])

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  #1*11*2048
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self,radius=0.03,k = 0.1):
        super(get_loss, self).__init__()
        self.radius = radius
        self.k      = k
    # def compute_normals(points):
    #     # 计算曲面法向量
    #     # 计算协方差矩阵
    #     covariance_matrix = np.cov(points, rowvar=False)
    #     # 计算协方差矩阵的特征向量
    #     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    #     # 找到最小特征值对应的特征向量作为法向量
    #     min_eigenvalue_index = np.argmin(eigenvalues)
    #     normal = eigenvectors[:, min_eigenvalue_index]
    #     return normal
    
    # def get_ball_points(pointCloud,center):
    #     random_radius = abs(np.random.normal(0,0.1))
    #     new_point_cloud_idx = []
    #     for idx in range(len(pointCloud)):
    #         # 计算点与球心的距离
    #         distance = np.linalg.norm(pointCloud[idx] - center)
    #         # 检查点是否在球的外部
    #         if distance > random_radius:
    #             new_point_cloud_idx.append(idx)
    #     return new_point_cloud_idx
    
    # 应该在他认为的正向样本点上计算法向量并求与label相关的平面法向量之间的误差
    # 这样，反向样本的误差会更小，反向样本会依沿着更小前进。
    # 输入应该是BatchSize * Point个数* 3的大小,GTnormals应该也是一样的。
    def get_normal(self, estimated_points,groundTruth_normals):
        pcd_estimated = o3d.geometry.PointCloud()
        pcd_estimated.points = o3d.utility.Vector3dVector(np.asarray(estimated_points[:,:3].cpu()))
        pcd_estimated.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, max_nn=30))
        # 万一有些是没有算出来normals，所以为了防止错误计算，就是normal的idx不对应point的idx，所以采用idxlist进行迭代
        tmp = [] #存法向量存在的points的索引
        for idx in range(len(pcd_estimated.points)):
            if np.isnan(np.asarray(pcd_estimated.normals[idx]).all()) != False:
                tmp.append(idx)
        norm_sum_error = 0
        for idx in tmp:
            vec1 = pcd_estimated.normals[idx]
            vec2 = groundTruth_normals[idx]
            cosine_theta = np.dot(vec1, vec2)/(norm(vec1) * norm(vec2))
            cosine_error = cosine_theta*cosine_theta
            cosine_error = 1 - cosine_error
            norm_sum_error += cosine_error
        return norm_sum_error
    
    #输入为pred = 65536*11 target = 65536*1
    #新增输入的points normals必须为 BatchSize*Points*3
    def forward(self, pred, target,points,normals,trans_feat):#,weight):
        norm_error_sum = 0
        
        for i in range(len(points)):
            norm_error_sum += self.get_normal(points[i][:][:],normals[i][:][:])

        total_loss = (1-self.k)*F.nll_loss(pred, target)+ self.k*norm_error_sum#,weight=weight) #+ sum

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)     
    (model(xyz))