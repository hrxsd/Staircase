# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import numpy as np
#######数据集+模型输出 without ICP 测试导入###############
np.set_printoptions(threshold=np.inf)
import sys
sys.path.append("D:\pytorch_project\pointnet2_pytorch-master")
from data_utils.mydataset import MyDataSet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path
from models.pointnet2_sem_seg_msg import get_model 
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import open3d as o3d
import random
from Lib.visualizelib import high_level_pcd
########################################################
def get_seg_pointcloud(rootpath,index,npoints=5000):
    #数据提取
    dataset = MyDataSet(root=rootpath,label=2,npoints=5000)
    points,_ = dataset.__getitem__(index) #2500*3+2500*1
    points_raw = points # 先保存
    
    #网络准备
    state_dict = torch.load('D:\\pytorch_project\\pointnet2_pytorch-master\\log\\sem_seg\\segmsg_datasetl3_normalize_rrotate\\checkpoints\\best_model.pth')
    classifier = get_model(2).cuda()
    classifier.load_state_dict(state_dict['model_state_dict'])
    #classifier.load_state_dict(state_dict['model_state_dict'], False)
    classifier.eval()
    #网络输出
    points = points.transpose(1, 0).contiguous()#3*2500
    points = points.view(1, points.size()[0], points.size()[1]).cuda()#reshape batchsize为1，第二维度为3 第三维度2500 
    pred,_  = classifier(points) #1*2500*2矩阵   
    #pred   = list(pred)
    pred_choice = pred.data.max(2)[1]
    return pred, pred_choice, points_raw

# 返回pointnet++输出中的点云的labelled_index
def viz_pointcloud_o3d(labelled_points,display = False):#输入点云 输出点云显示
    pcd_vector = o3d.geometry.PointCloud() #生成Pointcloud对象，o3d需要
    colors = np.zeros((5000,3)) # 颜色矩阵
    labelled_index = []#保存标签的index
    for j in range(len(labelled_points)): #颜色根据label变化
        if labelled_points[j][3] == 1:
            colors[j] = [1,0,0] # RGB 确定为红色
            labelled_index.append(j)
        elif labelled_points[j][3] == 0:
            colors[j] = [0,0,1] # RGB确定为蓝色
        else:
            pass
    if display == True:
        coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
        pcd_vector.colors = o3d.utility.Vector3dVector(colors)
        pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])
        o3d.visualization.draw_geometries([pcd_vector]+[coord], window_name="pointnet++结果输出",point_show_normal=False,width=800,  height=600)  
    return labelled_index    


if __name__ == '__main__':
    for i in range(70):
        rootpath = 'D:\pytorch_project\StairsSet\downsampled_label3'
        _,pred_choice,points_raw = get_seg_pointcloud(rootpath,i)
        
        # 拼接预处理
        points_raw = points_raw.cuda() #原先在cpu
        pred_choice = pred_choice.view(-1,1) #2500个一维数组变成 2500*1的二维张量
        labelled_points = torch.cat((points_raw,pred_choice),1) # 拼接语义点云
        labelled_points = labelled_points.cpu().numpy()#转cpu 转numpy

        viz_pointcloud_o3d(labelled_points,display=True)

