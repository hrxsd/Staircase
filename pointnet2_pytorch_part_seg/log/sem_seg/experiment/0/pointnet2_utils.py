import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()
#归一化点云，得出以centroid为中心的坐标，球半径为1的点云
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# 该函数用来在ball query过程中确定每一个点距离采样点的欧几里得距离。
def square_distance(src, dst):
    # 函数输入是两组点，N为第一组点的个数，M为第二组点的个数，C为输入点的通道数（如果是xyz时C=3）.
    # 返回的是两组点之间两两的欧几里德距离，即N × M 的矩阵。由于在训练中数据通常是以Mini-Batch的形式输入的，
    # 所以有一个Batch数量的维度为B。
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]   [Batchsize,Number of Points, channels]
        dst: target points, [B, M, C]   [Batchsize,Number of Points, channels]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))#[B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1) #dim=-1 是最高维上加
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# 最远点采样是Set Abstraction模块中较为核心的步骤，从输入点云中按所需点个数npoint采样出点云子集，且点点之距要足够远。
# 合格函数是第一步，其返回结果是npoint个采样点在原始点云中的索引。该结果用来做球采样
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples ！！！
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    #初始化一个centroid矩阵，用于存储npoint个采样点的索引位置，大小为B*npoint 因为是索引 所以是npoint维。
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    
    #distance矩阵(B*N)记录某个batch中所有点距离某一个点的距离，初始化的值很大，后面会迭代更新。
    distance = torch.ones(B, N).to(device) * 1e10
    
    #farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    #batch indices 初始化为0~(N-1)的数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    #直到采样点达到npoint，否则进行以下迭代
    for i in range(npoint):

        # 设当前的采样点centroid为当前的最远点farthest
        centroids[:, i] = farthest
        
        # 取出这个centroid的坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        # 计算点集中的所有点到这个centroid的欧式距离        
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的值，则更新distance矩阵的对应值
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.max(distance, -1)[1]
    return centroids

# 按照输入的点云数据和索引返回由索引的点云数据。
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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# 用于寻找球形领域中的点。输入中radius为球形领域的半径，nsample为每个领域中要采样的最多的点，
# new_xyz为S个球形领域的中心（由FPS在前面得出的centroid），xyz为所有的点云；
# 输出为每个样本的每个球形领域的nsample个采样点集[B,S,nsample]的索引
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape     #[B,N,C]
    _, S, _ = new_xyz.shape #[B,S,C]
    # arange(start=0,end,step=1) 返回0到N-1之间数构成的的一维张量 再复制为 B*S*N个 
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    #常数和tensor比较大小后作为tensor索引
    group_idx[sqrdists > radius ** 2] = N # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]# 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N # 找到group_idx中值等于N的点
    group_idx[mask] = group_first[mask]    # 将这些点的值替换为第一个点的值
    return group_idx

# Sampling + Grouping主要用于将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征。
# 需要用到上面定义的那些函数，分成了sample_and_group和sample_and_group_all两个函数，
# 其区别在于sample_and_group_all直接将所有点作为一个group。
'''
先用farthest_point_sample函数实现最远点采样FPS得到采样点的索引，再通过index_points将这些点的从原始点中挑出来，作为new_xyz
利用query_ball_point和index_points将原始点云通过new_xyz 作为中心分为npoint个球形区域其中每个区域有nsample个采样点
每个区域的点减去区域的中心值
如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
sample_and_group_all直接将所有点作为一个group，即增加一个长度为1的维度而已，当然也存在拼接新的特征的过程，这里不再细述。
'''
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: 做FPS的个数 
        radius: 球采集的半径
        nsample: 每一个球采集的点的个数
        xyz: 点的位置的旧特征, [B, N, C]  这俩位置信息变特征信息
        points: 点数据的新特征, [B, N, D]
    Return:
        new_xyz: 采样的点的位置信息  [B, npoint, C]
        new_points: 采样的点的信息, [B, npoint, nsample, C+D]
    """

    B, N, C = xyz.shape #他没有了pointnet的dataset类的随机采样
    S = npoint #npoint 个球形中心点用来采样 与new_xyz的行数一样

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint] FPS返回的是采样的npoint的点的index
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)         # 采样结束后的得出的[B,S,C] fps_index对应的点云子集
    torch.cuda.empty_cache()
    #球大小，每个球中采集多少个点，原始点集，fps采样点集，
    idx = query_ball_point(radius, nsample, xyz, new_xyz) #返回值是输出为每个样本的每个球形领域的nsample个采样点集[B,S,nsample]的索引
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)# 以球中心点为原点。
    torch.cuda.empty_cache()

    #拼接点的坐标和特征
    if points is not None:
        grouped_points = index_points(points, idx)  #cat是拼接
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points #返回FPS采样得出的点 和 变过参考系的球采样得出的点

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# 首先先通过sample_and_group的操作形成局部的group，Sampling + Grouping主要用于将整个点云分散成局部的group，然后对局部的group中的每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征。
# SetAbstraction 以N*(d+C)维输入，以N'*(d+C')维输出。N N'都是点的个数，d维为三维坐标 C维为特征向量维数。

class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint # FPS采样得出点的个数
        self.radius = radius # 球采样半径
        self.nsample = nsample # 每个球采样的内部点的个数
        '''
        它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面。但不同于一般的 list，
        加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。
        '''
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        #初始化网络结构的好办法 
        for out_channel in mlp: # mlp: mlp输入输出通道数的列表,如[64, 64, 128]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1)) #in_channels out_channels kernelsize
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data,       [B, D, N] 
        Return:
            new_xyz: sampled points position data, [B, C, S] 
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) #变B,N,C
        if points is not None:
            points = points.permute(0, 2, 1) #变 B,N,D
        
        # 形成group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else: #返回FPS采样得出的点 和 变过参考系的球采样得出的点
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        # 以下是pointnet操作，对局部group中的每一个点做MLP
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，对[C+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对每个group做一个max pooling得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

#pointnetsetabstractionMSG类实现MSG方法，这里radius_list输入一个list, eg=[0.1,0.2,0.4]
#对于不同的半径做ball query，将不同半径之下的点云特征保存在new_points_list中，最后拼接在一起
class PointNetSetAbstractionMsg(nn.Module): 
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)): #一共有两个mlplist元素
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]: #[[16, 16, 32], [32, 32, 64]]
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # B N C 
        if points is not None:
            points = points.permute(0, 2, 1) #B N D

        B, N, C = xyz.shape
        S = self.npoint #1024
        #最远点采样
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        #将不用半径点云特征保存在new_point_list
        new_points_list = []
        for i, radius in enumerate(self.radius_list):  #[0.05, 0.1]
            K = self.nsample_list[i] #[16, 32]
            #query ball point函数用于寻找球星领域中的点
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # 按照输入的点云数据和索引返回索引的点云数据
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            #if points is not None:
            grouped_points = index_points(points, group_idx)
            #拼接点特征和点坐标数据
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) #合并了
            #else:
            #    grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            #最大池化
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points) #不同半径下点云特征的列表

        new_xyz = new_xyz.permute(0, 2, 1)
        #拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

# 用于点云分割时，
# FP的实现主要通过线性插值和MLP完成
# 当点的个数只有一个的时候，采用repeat直接复制成N个点
# 当点的个数大于一个的时候，采用线性插值上采样
# 拼接上下采样对应点的SA层的特征，再对拼接后的每一个点都做一个MLP。
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:#当点的个数只有一个的时候，repeat成N个点
            interpolated_points = points2.repeat(1, N, 1)
        else:#线性插值部分
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)#距离越远的点权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm#对于每一个点的权重再做一个全局的归一化
            #获得插值点
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            #拼接上下采样前对应点SA层的特征
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        #对拼接后的每一点都做一个MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

