#2023年4月5日16点52分
import torch.utils.data as data
import os
import json
import numpy as np
import torch
import sys
sys.path.append(r"D:\\Project_on_going\\pointnet2_pytorch_part_seg")
from data_utils.data_augmentation import *

###############################################
#功能：数据对齐
#输入：原始点云、pred choice
#输出：拼接好的点云
############################################### 
def data_align(__points_raw,__pred_choice):
    __pred_choice = __pred_choice.reshape(-1,1) #N个一维数组变成 N*1的二维张量
    __output = np.concatenate((__points_raw,__pred_choice),1) # 拼接语义点云
    #__output = __output.numpy()
    return __output

###############################################
#功能：
#输入：
#输出：
############################################### 
# 返回pointnet++输出中的点云的labelled_index
def viz_pointcloud_o3d(labelled_points,windows_name,display = False):#输入点云 输出点云显示
    __pcd_vector = o3d.geometry.PointCloud() #生成Pointcloud对象，o3d需要
    __colors = np.zeros((len(labelled_points),3)) # 颜色矩阵
    labelled_index = []#保存标签的index
    for j in range(len(labelled_points)): #颜色根据label变化
        if labelled_points[j][3] != 10:
            __colors[j] = [1,0,0] # RGB 确定为红色
            labelled_index.append(j)
        else:
            __colors[j] = [0,0,1] # RGB确定为蓝色
        
    if display == True:
        coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
        __pcd_vector.colors = o3d.utility.Vector3dVector(__colors)
        __pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])
        o3d.visualization.draw_geometries([__pcd_vector]+[coord], window_name=windows_name,
                                        point_show_normal=False,
                                        width=800,height=600,left = 1500,top=500)  
    return labelled_index 

def viz_pointcloud_o3d_for_rrb_debug(labelled_points,windows_name,open3d_object,rrb_sampling_center,display = False):#输入点云 输出点云显示
    __pcd_vector = o3d.geometry.PointCloud() #生成Pointcloud对象，o3d需要
    __colors = np.zeros((len(labelled_points),3)) # 颜色矩阵
    labelled_index = []#保存标签的index
    for j in range(len(labelled_points)): #颜色根据label变化
        if labelled_points[j][3] != 10:
            __colors[j] = [1,0,0] # RGB 确定为红色
            labelled_index.append(j)
        else:
            __colors[j] = [0,0,1] # RGB确定为蓝色
    
    center = o3d.geometry.PointCloud()
    center_color = [0,0,0]
    center.colors = o3d.utility.Vector3dVector(np.asarray(center_color[:3]).reshape(-1,3))
    center.points = o3d.utility.Vector3dVector(np.asarray(rrb_sampling_center).reshape(-1,3))
    if display == True:
        coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
        __pcd_vector.colors = o3d.utility.Vector3dVector(__colors)
        __pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])
        o3d.visualization.draw_geometries([__pcd_vector]+[coord]+[open3d_object]+[center], window_name=windows_name,
                                        point_show_normal=False,
                                        width=800,height=600,left = 1500,top=500)  
    return labelled_index 

def count_files_in_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort(key=lambda x: int(os.path.basename(x)))

    result = {}
    for subfolder in subfolders:
        file_count = len(os.listdir(subfolder))
        result[os.path.basename(subfolder)] = file_count

    return result

class Train_DataSet(data.Dataset):
    def __init__(self, root, npoints, rrns_prob, rrb_radius_max, rgm_step_divetion, 
                 rrns_enable,rrb_enable,rrs_enable,rgm_enable,rgrp_enable,rspo_eanble):
        
        #导入外参
        self.npoints = npoints
        self.rrns_prob = rrns_prob
        self.rrb_radius_max = rrb_radius_max
        #self.rrs_step = rrs_step
        self.rgm_step_divetion = rgm_step_divetion
        #导入数据增强开启开关
        self.rrns_enable = rrns_enable
        self.rrb_enable  = rrb_enable
        self.rrs_enable = rrs_enable
        self.rgm_enable = rgm_enable
        self.rgrp_enable = rgrp_enable
        self.rspo_eanble = rspo_eanble


        #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
        self.datapath = {} #文件序数和文件完整路径对应字典
        self.datapath_norms = {}
        self.root = root
        self.dictNumber = 10
        #self.dictNumber_norms = 10
        #TODO
        
        self.fileslist = []
        
        #self.fileslist.append(0)
        file_counts = count_files_in_subfolders(root)
        for k,v in file_counts.items():
            self.fileslist.append(v)
        # fileslist = [22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
        self.t = 0
        self.tmp = []
        self.tmp.append(0)
        for k,v in file_counts.items():
            self.t += v
            self.tmp.append(self.t)
        # tmp = [0, 22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
        #字典赋值
        #法向量标注文件查找字典的赋值
        for dictOrder in range(self.dictNumber): #dictOrder 0,1,2,3,4,5,6,7,8,9
            #tmp = self.fileslist[dictOrder] 
            for filesOrder in range(0,self.fileslist[dictOrder]):#filesOrder 0到当前dictorder的长度
                self.datapath[filesOrder+self.tmp[dictOrder]] = os.path.join(self.root,str(dictOrder+1),'{}_{}.txt'.format(dictOrder+1,filesOrder+1))
        
        #字典测试
        #print(self.datapath)
        
    def __getitem__(self, index):        

        pointSet     = np.loadtxt(str(self.datapath[index]),usecols=(0,1,2)).astype(np.float32)     
        pointLabel   = np.loadtxt(str(self.datapath[index]),usecols=(6)).astype(np.int64)           
        pointNormal  = None
        try:
            pointNormal = np.loadtxt(str(self.datapath[index]),usecols=(7,8,9)).astype(np.float32)
        except IndexError:
            print("pointNormal doesn't exist")
            pass

        # # 检查数据合规范要求 
        # for i in np.nditer(pointLabel):
        #     if pointLabel[i] >10 or pointLabel[i]<0:
        #         print("datapath:",self.datapath[index])
        #         print("row{}",format(i))
        #         raise Exception("标记数据不合规")
        

        # RGRP
        # 随机调整各类别点云个数
        if self.rgrp_enable == True:
            pointSet,pointLabel = random_gaussian_remove_point(point=pointSet,label=pointLabel)       
        
        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示RGRP效果',True)

        # RRS
        # TODO添加一个可视化
        # 随机去掉某N个台阶
        if self.rrs_enable == True:
            labelset = set(pointLabel)
            labelset_copy = labelset.copy()
            if len(labelset_copy) > 3: #数据大于3包含有10和其他至少2个楼梯
                labelset_copy.remove(10)
                pointLabelClass = max(labelset_copy)
                remove_step_num1 = random.randint(0,2)
                randomly_removed_steps_list = random_remove_steps(pointLabelClass,remove_step_num1)#大于2你才能考虑去掉2个
            elif len(labelset_copy) == 3:#数据等于3包含有10和其他2个楼梯
                labelset_copy.remove(10)
                pointLabelClass = max(labelset_copy)
                remove_step_num2 = random.randint(0,1)
                randomly_removed_steps_list = random_remove_steps(pointLabelClass,remove_step_num2)
            else:#数据只包含一个楼梯和10 || 10 则不去掉
                randomly_removed_steps_list = []
            #转成集合
            randomly_removed_steps_set = set(randomly_removed_steps_list)
            complement_set = labelset - randomly_removed_steps_set
            complement_list = list(complement_set) #补集转列表
            choice_idx_rand_rm_steps = []
            for i in range(len(pointLabel)):
                if pointLabel[i] in complement_list:
                    choice_idx_rand_rm_steps.append(i)
            pointSet   = pointSet[choice_idx_rand_rm_steps, :] 
            pointLabel = pointLabel[choice_idx_rand_rm_steps]
            try:
                pointNormal = pointNormal[choice_idx_rand_rm_steps,:]
            except NameError:
                pass

        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示RRS效果',True)

        # RRNS
        # 随机去掉非楼梯点 概率为30.85% 
        # TODO 这里需要考虑到纯10和10+1 10+2
        # BUG记录：这里有个判断变量是提取了labelset的长度，万一在上面把楼梯数量去
        if self.rrns_enable and len(set(pointLabel)) >= 2:
            remove_notstep_flag = random_remove_not_step_points(self.rrns_prob)
            def find_indexes(arr, value):#找特定值的index
                return [index for index, element in enumerate(arr) if element != value]
            if remove_notstep_flag == 1:
                labelset = set(pointLabel)
                if len(labelset) > 1: #数据大于1包含有10和其他至少1个楼梯
                    choice_idx_rand_rm_notsteps = find_indexes(list(pointLabel), 10)
                    pointSet   = pointSet[choice_idx_rand_rm_notsteps, :] 
                    pointLabel = pointLabel[choice_idx_rand_rm_notsteps]
                    try:
                        pointNormal = pointNormal[choice_idx_rand_rm_notsteps,:]
                    except NameError:
                        pass
                else:#数据只包含10 
                    pass
            else:
                pass
        
        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示RRNS效果',True)


        # RRB
        # 随机挖掉一个球
        if self.rrb_enable and len(set(pointLabel)) >= 2:
            def find_indexes(arr, value):#找特定值的index
                return [index for index, element in enumerate(arr) if element != value]
            idx_list_for_center_computing = find_indexes(list(pointLabel),10)
            PC_center_computing = pointSet[idx_list_for_center_computing,:]
            PC_center_computing = np.mean(PC_center_computing,0)

            choice_idx_ball_sampling,rrb_sampling_radius,rrb_sampling_center= random_remove_ball_points(pointSet,pointLabel,PC_center_computing,self.rrb_radius_max)
            pointSet   = pointSet[choice_idx_ball_sampling,:] 
            pointLabel = pointLabel[choice_idx_ball_sampling]
            try:
                pointNormal = pointNormal[choice_idx_ball_sampling,:]
            except NameError:
                pass

        #     mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=rrb_sampling_radius,resolution=100)
        #     mesh_sphere.compute_vertex_normals()
        #     mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        #     # 将Mesh的中心设置为你要放大显示的点的位置
        #     mesh_sphere.translate(np.asarray(rrb_sampling_center).reshape(3,-1))
        #     pc_colored = data_align(pointSet,pointLabel)
        #     _ = viz_pointcloud_o3d_for_rrb_debug(pc_colored,'显示RRB效果',mesh_sphere,rrb_sampling_center,True)

        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示最终效果',True)

        #Pointnet++的随机降采样算法
        try:
            choice = np.random.choice(len(pointLabel), self.npoints, replace=True)# 重新采样到self.npoints个点
            choice = np.asarray(choice)
            pointSet   = pointSet[choice, :] #resample
            pointLabel = pointLabel[choice]
            try:
                pointNormal = pointNormal[choice,:]
            except NameError:
                pass
        except:
            pass

         # 去中心化 # 归一化#计算到原点的最远距离
        normalization_center = np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        pointSet = pointSet - normalization_center # center
        try:
            dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
            pointSet = pointSet / dist #scale
        except ValueError:
            print("Value Error Ocurrs:",self.datapath[index])
            raise Exception
       
        
        # RGM
        # 随机高斯移动
        if self.rgm_enable:
            pointSet = random_guassian_move(pointSet,self.rgm_step_divetion)      

        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示RGM效果',True)

        # RSPO
        # 随机打乱点云排列
        if self.rspo_eanble == True:
            pointSet,pointLabel,pointNormal = random_scramble_point_order(points=pointSet,labels=pointLabel,pointsnorm=pointNormal)
        
        # pc_colored = data_align(pointSet,pointLabel)
        # _ = viz_pointcloud_o3d(pc_colored,'显示RSPO效果',True)

        pointSet   = torch.from_numpy(pointSet)
        pointLabel = torch.from_numpy(pointLabel)

        return pointSet,pointNormal,pointLabel

    def __len__(self):
        return len(self.datapath)

class Eval_DataSet(data.Dataset):
    def __init__(self, root, npoints):
        
        #导入外参
        self.NPOINT_LIST = [1024,2048,4096,8192]
        self.npoints = npoints

        
        #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
        self.datapath = {} #文件序数和文件完整路径对应字典
        self.datapath_norms = {}
        self.root = root
        self.dictNumber = 10
        #self.dictNumber_norms = 10
        #TODO
        
        self.fileslist = []
        
        #self.fileslist.append(0)
        file_counts = count_files_in_subfolders(root)
        for k,v in file_counts.items():
            self.fileslist.append(v)
        # fileslist = [22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
        self.t = 0
        self.tmp = []
        self.tmp.append(0)
        for k,v in file_counts.items():
            self.t += v
            self.tmp.append(self.t)
        # tmp = [0, 22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
        #字典赋值
        #法向量标注文件查找字典的赋值
        for dictOrder in range(self.dictNumber): #dictOrder 0,1,2,3,4,5,6,7,8,9
            #tmp = self.fileslist[dictOrder] 
            for filesOrder in range(0,self.fileslist[dictOrder]):#filesOrder 0到当前dictorder的长度
                self.datapath[filesOrder+self.tmp[dictOrder]] = os.path.join(self.root,str(dictOrder+1),'{}_{}.txt'.format(dictOrder+1,filesOrder+1))
        
        #字典测试
        #print(self.datapath)
        
    def __getitem__(self, index):        
        
        #print(self.datapath[index])
        pointSet     = np.loadtxt(str(self.datapath[index]),usecols=(0,1,2)).astype(np.float32)     # 读取点云
        pointLabel   = np.loadtxt(str(self.datapath[index]),usecols=(6)).astype(np.int64)           # 读取标签1
        pointNormal  = None
        try:
            # 可能引发异常的代码块
            pointNormal = np.loadtxt(str(self.datapath[index]),usecols=(7,8,9)).astype(np.float32)
        except IndexError:
            # 异常处理代码块
            print("pointNormal doesn't exist")
            pass
        
        #Pointnet++的随机降采样算法
        try:
            choice = np.random.choice(len(pointLabel), self.npoints, replace=True)# 重新采样到self.npoints个点
            choice = np.asarray(choice)
            pointSet   = pointSet[choice, :] #resample
            pointLabel = pointLabel[choice]
            try:
                pointNormal = pointNormal[choice,:]
            except NameError:
                pass
        except:
            pass

         # 去中心化 # 归一化#计算到原点的最远距离
        pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale

        pointSet   = torch.from_numpy(pointSet)
        pointLabel = torch.from_numpy(pointLabel)
        try:
            pass
            #pointNormal = torch.from_numpy(pointNormal)
        except NameError:
            pass
        return pointSet,pointNormal,pointLabel

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser('Model') #生成parser对象的方法
        parser.add_argument('--model',  type=str,           default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
        parser.add_argument('--learning_rate',type=float,   default=0.001,              help='Initial learning rate [default: 0.001]')
        parser.add_argument('--gpu', type=str,              default='0',                help='GPU to use [default: GPU 0]')
        #parser.add_argument('--optimizer', type=str,        default='Adam',             help='Adam or SGD [default: Adam]')
        parser.add_argument('--log_dir', type=str,          default=None,               help='Log path [default: None]') #可以指定log的文件夹名
        parser.add_argument('--decay_rate', type=float,     default=1e-4,               help='weight decay [default: 1e-4]')
        parser.add_argument('--step_size', type=int,        default=20,                 help='Decay step for lr decay [default: every 10 epochs]')
        parser.add_argument('--lr_decay', type=float,       default=0.7,                help='Decay rate for lr decay [default: 0.7]')
        parser.add_argument('--test_area', type=int,        default=5,                  help='Which area to use for test, option: 1-6 [default: 5]')
        parser.add_argument('--trian_dataset',type=str,     default='D:\Project_on_going\pointnet2_pytorch_part_seg\dataset\dataset_withlabell_trian_downsampled_pointnormal',help='train dataset root path')
        parser.add_argument('--eval_dataset',type=str,      default='D:\Project_on_going\pointnet2_pytorch_part_seg\dataset\dataset_withlabell_eval_downsampled_pointnormal',help='eval dataset root path')

        parser.add_argument('--epoch', type=int,            default=200,                help='Epoch to run [default: 32]')
        parser.add_argument('--npoint', type=int,           default=2048,               help='Point Number [default: 4096]')
        parser.add_argument('--batch_size', type=int,       default=32,                 help='Batch Size during training [default: 16]')
        parser.add_argument('--rrns_prob',type=float,       default=0.5,             help='radius for caculating norms of loss')
        parser.add_argument('--rrb_radius',type=float,      default=0.1,             help='radius for caculating norms of loss')
        parser.add_argument('--rgm_step',type=float,        default=0.002,            help='radius for caculating norms of loss')
        parser.add_argument('--loss_k',type=float,          default=0.1,              help='loss互补系数')
        parser.add_argument('--loss_radius',type=float,     default=0.03,             help='radius for caculating norms of loss')
        
        parser.add_argument('--newloss_enable',type=bool, default=True,               help='newloss_enable')

        parser.add_argument('--rrns_enable',type=bool, default=True,               help='random remove non steps')
        parser.add_argument('--rrb_enable',type=bool,default=True,                 help='random remove ball')
        parser.add_argument('--rrs_enable',type=bool, default=True,                help='random remove steps')
        parser.add_argument('--rgm_enable',type=bool, default=True,                help='random gaussion move')
        parser.add_argument('--rgrp_enable',type=bool,default=True,                help='random gaussion remove point')
        parser.add_argument('--rspo_eanble',type=bool, default=True,               help='random scramble point order')
    
        return parser.parse_args()
    
    args = parse_args()
    TRAIN_DATASET = Train_DataSet(root = args.trian_dataset,
                                npoints = args.npoint,
                                rrns_prob = args.rrns_prob,
                                rrb_radius_max = args.rrb_radius,
                                rgm_step_divetion = args.rgm_step,
                                rrns_enable = args.rrns_enable,
                                rrb_enable = args.rrb_enable,
                                rrs_enable = args.rrs_enable,
                                rgm_enable = args.rgm_enable,
                                rgrp_enable = args.rgrp_enable,
                                rspo_eanble = args.rspo_eanble)
    for j in range(100):
        for i in range(550):
            data = TRAIN_DATASET.__getitem__(i)
            print(i)