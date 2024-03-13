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

class get_loss_new(nn.Module):
    def __init__(self,radius=0.03,k = 0.1):
        super(get_loss_new, self).__init__()
        self.radius = radius
        self.k      = k
    
    #############################################################################
    # 功能：从seg_pred_has_batch中提取出各类正向预测样本的index，并从points中提取出对应的点按BatchSize*Step*Points*3的格式返回
    # 输入：原始点云|网络预测结果(这个会越来越好，最开始是0到10都有的)
    # 注意：考虑在后期加入
    # Input: points:             Tensor in GPU, 32*3*2048
    #      : seg_pred_has_batch: Tensor in GPU, 32*2048*11
    # Output: points:            List, BatchSize*StepNum*PointsNum*3
    ###############################################################################
    def extract_points_from_pred_labal(self,points,seg_pred_has_batch,BATCHSIZE,non_choosen_category = 10):
        
        # 预处理
        
        #   1.预整形:变回B*N*C的Tensor
        points = points.transpose(2,1) 
        
        #   2.提取pred_label
        pred_label_of_batches_list = [] #BatchSize*PointsNum
        for batch in range(BATCHSIZE):     
            pred_label_of_batches_list.append(list(seg_pred_has_batch[batch].cpu().data.max(1)[1].numpy())) 

        #   功能-下面使用的就是points和pred_label_of_batches_list
        #   3.根据pred_label_of_batches_list提取数据
        pred_label_species_of_batches_list = [] # BatchSize*Steps，第二维度是预测标签种类维度
        for batch in range(BATCHSIZE):
            pred_label_species_of_batches_list.append(list(set(pred_label_of_batches_list[batch])))

        #   4.从预测结果提取index并提取对应points
        p = []  # BatchSize*StepNum*PointsNum*3
        for batch in range(BATCHSIZE):
            # TODO 这个初始化在没有10的情况下还对吗？
            # 之前的初始化方法： for _ in range(max(pred_label_array_of_batchs_list[batch])+1)
            tmp_p = [[],[],[],[],[],[],[],[],[],[],[]] #StepOrderNum*PointsNum*3
            # np.where doesn't work on list objects
            category_indices = np.reshape(np.argwhere(np.asarray(pred_label_of_batches_list[batch]) != non_choosen_category),(-1))
            for j in category_indices: # 每一个Batch中所有楼梯点的index
                tmp_p[pred_label_of_batches_list[batch][j]].append(list(points[batch][j][:].cpu().data.numpy()))
            p.append(tmp_p)
        return p
    
    #################################################################################
    # 功能：根据 target_has_batch 和 point_normal_gt 求出真实楼梯平均法向量
    # 输入：target_has_batch: BatchSize*points           Tensor
    #      point_normal_gt:  BatchSize*Point*3             List
    # 输出：step_avr_normal: BatchSzie * Step * 3         List
    # 注意：Batch级别操作，一个BatchSize只用运行一次
    #################################################################################
    def get_step_avr_normal(self,target_has_batch,point_normal_gt,BATCHSIZE):
        
        point_normal_gt = point_normal_gt.tolist()
        #1.从target_has_batch 求出每一个Batch中楼梯的S个楼梯点的index，
            #内部函数，作用于batch内
            #input: target Tensor 2048
            #output:tmp    List   Steps*idx_num
        def get_step_points_idx_list(target):
            category_indices = np.reshape(np.argwhere(target.cpu().numpy() != 10),(-1))
            tmp = [[],[],[],[],[],[],[],[],[],[],[]]
            for idx in category_indices:
                tmp[target[idx]].append(idx)
            return tmp
        
        tmp_step_idx_list_of_a_batch = [] #batch*step*points_index
        for stair in range(BATCHSIZE):    
            tmp_step_idx_list_of_a_batch.append(get_step_points_idx_list(target_has_batch[stair]))
        
        #TODO 调试到这里  出现了空问题，猜想是因为10号没有考虑。
        #2.从point_normal_gt根据上述index提取出楼梯点的normals
        #输入：point_normal_gt：BatchSize*Point*3 | tmp_step_idx_list_of_a_batch：BatchSize*step*points_index
        #TODO 这里多出来一个维度
        #输出：tmp_norm_list_of_a_batch = []：BatchSize * step_num * 1 * norms_num * 3 
        tmp_norm_list_of_a_batch = [] #32 * step_num * norms_num * 3
        for stair in range(BATCHSIZE):
            #################################################################
            tmp_norm_list_of_a_stair = [[],[],[],[],[],[],[],[],[],[],[]] #step_num * norms_num * 3
            for step in range(len(tmp_step_idx_list_of_a_batch[stair])):   
                tmp_norm_list_of_a_step = []
                for idx in range(len(tmp_step_idx_list_of_a_batch[stair][step])):
                    point_normal_xyz = point_normal_gt[stair][idx][:]
                    tmp_norm_list_of_a_step.append(point_normal_xyz)
                tmp_norm_list_of_a_stair[step].append(tmp_norm_list_of_a_step)
            ##################################################################
            tmp_norm_list_of_a_batch.append(tmp_norm_list_of_a_stair)
        
        #3.求平均法向量
        #输入：tmp_norm_list_of_a_batch = []：BatchSize * step_num * 1 * norms_num * 3
        #输出：tmp_avr_norm_for_a_batch = []：Batchsize * step_num * 3
        tmp_avr_norm_for_a_batch = [] # ：Batchsize * step_num * 3
        for stair in range(BATCHSIZE):
            tmp_norm_for_a_stair = [[],[],[],[],[],[],[],[],[],[],[]] # step_num * 3
            ############################################################################################
            for step in range(len(tmp_norm_list_of_a_batch[stair])): #其实就是11
                tmp_norm_for_a_step = [[],[],[]] #3*1
                #判定是不是对的？
                points_for_norm_calc = tmp_norm_list_of_a_batch[stair][step][0][:]
                #TODO 这里假如全0 会报错，是全猜10造成的没有法向量，之后考虑异常处理
                try:
                    avr_norm_x = np.sum(np.asarray(points_for_norm_calc),axis=0)[0]/float(len(points_for_norm_calc))
                    avr_norm_y = np.sum(np.asarray(points_for_norm_calc),axis=0)[1]/float(len(points_for_norm_calc))
                    avr_norm_z = np.sum(np.asarray(points_for_norm_calc),axis=0)[2]/float(len(points_for_norm_calc))
                    tmp_norm_for_a_step[0] = avr_norm_x
                    tmp_norm_for_a_step[1] = avr_norm_y
                    tmp_norm_for_a_step[2] = avr_norm_z
                    tmp_norm_for_a_stair[step] = tmp_norm_for_a_step
                except IndexError:
                    tmp_norm_for_a_step[0] = 0
                    tmp_norm_for_a_step[1] = 0
                    tmp_norm_for_a_step[2] = 0
                    tmp_norm_for_a_stair[step] = tmp_norm_for_a_step

            ############################################################################################
            tmp_avr_norm_for_a_batch.append(tmp_norm_for_a_stair)

        batch_steps_avr_normals = tmp_avr_norm_for_a_batch
        return batch_steps_avr_normals

    #######################################################################
    # 功能：在预测正向点上计算出法向量；这样，反向样本的误差会更小，反向样本会依沿着更小前进。
    # 输入：Steps*N'*3 预测正向点
    # 输出：Steps*N'*3 预测正向点法向量
    # 注意：法向量个数有可能少于点的个数,这里需要对open3d的normal对象检查形状
    #######################################################################
    def get_pred_normal(self, pred_points):
        
        steps_num = len(pred_points)
        
        #steps*Points*3
        #全10就会空。
        pcd_estimated_norm_list = [[],[],[],[],[],[],[],[],[],[],[]] 
        for step in range(steps_num): 
            if(len(pred_points[step]) == 0):
                continue
            else:
                pcd_estimated_norm = o3d.geometry.PointCloud()
                pcd_estimated_norm.points = o3d.utility.Vector3dVector(pred_points[step])
                pcd_estimated_norm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, max_nn=30))
                pcd_estimated_norm_list[step] = pcd_estimated_norm.normals
        
        # 万一有些是没有算出来normals，所以为了防止错误计算，就是normal的idx不对应point的idx，所以采用idxlist进行迭代
        tmp_list = [[],[],[],[],[],[],[],[],[],[],[]] #第一维度是steps*具有normal的点的idx #存法向量存在的points的索引
        for step in range(steps_num):
            tmp = []
            if len(pcd_estimated_norm_list[step]) > 0:  
                for idx in range(len(pcd_estimated_norm_list[step])):
                    tmp.append(idx)
            else:
                continue
            tmp_list[step].append(tmp)
        
        # 因为只用算求出来的点，所以不用考虑求不出来的点，做假设很多点是能被求出来的。
        # 整形
        normals = []
        for step in range(steps_num):
            t = []
            for idx in range(len(tmp_list[step])):
                t.append(pcd_estimated_norm_list[step][idx])
            normals.append(t)
        
        return normals
    

    #################################################################################
    # 功能：获取每一个预测正向点的法向量跟真实平均法向量之间的误差
    # 输入：Steps*N'*3 预测正向点法向量 | Steps*3 真实楼梯平均法向量
    # 输出：cosine_error
    # 注意：要求steps维度个数对齐,这儿还把gt为000 是跳过给考虑了
    #################################################################################
    def get_normal_error(self,pred_normal,gt_normal):
        norm_sum_error = 0
        steps_num = len(pred_normal)
        for step in range(steps_num):
            points_num = len(pred_normal[step])
            for point in range(points_num):
                
                if np.linalg.norm(gt_normal[step][:]) == 0:
                    break
                else:
                    vec1 = pred_normal[step][point][:]
                vec2 = gt_normal[step][:]
                cosine_theta = np.dot(vec1, vec2)/(norm(vec1) * norm(vec2))
                cosine_error = cosine_theta*cosine_theta
                cosine_error = 1 - cosine_error
                norm_sum_error += cosine_error
            if np.linalg.norm(gt_normal[step][:]) == 0:
                continue
        return norm_sum_error
    


    #################################################################################
    #功能：计算误差，包含交叉熵误差+曲率误差
    #Input: seg_pred:               Tensor in GPU, 65536*11
    #       seg_pred_has_batch:     Tensor in GPU, 32*2048*11
    #       target:                 Tensor in GPU, 65536 
    #       target_has_batch:       Tensor in GPU, 32*2048
    #       points:                 Tensor in GPU, 32*3*2048
    #       points_gt_normal:       NumpyArray   , 32*2048*3
    #output: loss
    #################################################################################
    def forward(self, seg_pred, seg_pred_has_batch, 
                      target, target_has_bacth, 
                      points, points_gt_normal):#,trans_feat):#,weight):
        
        #内参计算
        BATCHSIZE = len(target_has_bacth)

        #误差变量
        norm_error_sum = 0
        
        #Test Passed !!!!
        #Input: points:             Tensor in GPU, 32*3*2048
        #     : seg_pred_has_batch: Tensor in GPU, 32*2048*11
        #output: points:            List,          BatchSize*StepNum*PointsNum*3         
        points = self.extract_points_from_pred_labal(points,seg_pred_has_batch,BATCHSIZE)
        
        #计算gt 平均台阶法向量
        #注意事项：求不出来的法向量 设置成000。
        #Input: target_has_batch:       Tensor in GPU, 32*2048*1
        #     : points_gt_normal:       NumpyArray   , 32*2048*3
        #output: steps_avr_normal:            List,    BatchSize*StepNum*3  
        steps_avr_normal = self.get_step_avr_normal(target_has_bacth,points_gt_normal,BATCHSIZE)
        
        #TODO debug到这个，求不出来的法向量 设置成000
        #计算每一个batch的误差求和
        
        for stair in range(BATCHSIZE): #Batch level computing
            #Input: points:  List,  BatchSize*StepNum*PointsNum*3
            #Output:  points_norm_of_a_stair: List, StepNum*PointsNum*3
            points_norm_of_a_stair = self.get_pred_normal(points[stair])
            norm_error_sum += self.get_normal_error(points_norm_of_a_stair,steps_avr_normal[stair])
        
        #总的误差计算
        total_loss = (1-self.k)*F.nll_loss(seg_pred, target)+ self.k*norm_error_sum#,weight=weight) #+ sum

        return total_loss

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):#,trans_feat):#,weight):
        total_loss = F.nll_loss(pred, target)#,weight=weight)
        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)     
    (model(xyz))