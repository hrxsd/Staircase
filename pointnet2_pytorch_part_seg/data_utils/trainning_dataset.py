#2023年4月5日16点52分
import torch.utils.data as data
import os
import json
import numpy as np
import torch

from data_utils.data_augmentation import *


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
        
        # RSPO
        # 随机打乱点云排列
        if self.rspo_eanble == True:
            pointSet,pointLabel = random_scramble_point_order(points=pointSet,labels=pointLabel)

        # RGRP
        # 随机调整各类别点云个数
        if self.rgrp_enable == True:
            pointSet,pointLabel = random_gaussian_remove_point(point=pointSet,label=pointLabel)
        
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

        # RRB
        # 随机挖掉一个球
        if self.rrb_enable:
            choice_idx_ball_sampling= random_remove_ball_points(pointSet,pointLabel,self.rrb_radius_max)
            pointSet   = pointSet[choice_idx_ball_sampling,:] 
            pointLabel = pointLabel[choice_idx_ball_sampling]
            try:
                pointNormal = pointNormal[choice_idx_ball_sampling,:]
            except NameError:
                pass
        
        # RRNS
        # 随机去掉非楼梯点 概率为30.85% 
        # TODO 这里需要考虑到纯10和10+1 10+2
        if self.rrns_enable:
            remove_notstep_flag = random_remove_not_step_points(self.rrns_prob)
            def find_indexes(arr, value):#找特定值的index
                return [index for index, element in enumerate(arr) if element == value]
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
        
        # RGM
        # 随机高斯移动
        if self.rgm_enable:
            pointSet = random_guassian_move(pointSet,self.rgm_step_divetion)      


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
    root = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled'
    npoints = 2048
    dataset = MyDataSet(root=root,npoints=npoints)
    data = dataset.__getitem__(1)
    print(data)
