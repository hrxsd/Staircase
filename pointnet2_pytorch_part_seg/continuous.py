
###必须的模块###
import numpy as np
from time import time

# import open3d.geometry

np.set_printoptions(threshold=np.inf)
from collections import  Counter
import sys
# sys.path.append("D:\pytorch_project\pointnet2_pytorch-master")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path

###网络相关###
from models.pointnet2_sem_seg_msg import get_model 
import torch
import torch.nn.parallel
import torch.utils.data
# from torch.autograd import Variable
###数据相关###
from mydataset_for_SAL import MyDataSet
# from data_utils.data_preprocess import compute_steps_num

###显示相关###
import open3d as o3d
from Lib.visualizelib import *

###最远点采样###
from models.pointnet2_utils import *

import readline
import threading

def get_pred_pointcloud(dataset_path,index,npoints):
        
    #外参
    MODEL_ROOT = './log/sem_seg/downsampled'
    MODEL_VERSION = 'rrs-rrns-rrb'
    
    MODEL_FINALPATH = r'checkpoints/best_model.pth'
    MODEL_PATH = os.path.join(MODEL_ROOT,MODEL_VERSION,MODEL_FINALPATH)
    ENV_MODEL = torch.device('cpu')
    NUM_CLASSES = 11
        
    #数据提取
    dataset = MyDataSet(root=dataset_path,npoints = npoints)
    # points,pointslabel = dataset.__getitem__(index) #npoints*3+npoints*1
    points,RGB_INFO = dataset.__getitem__(index)
    points_raw = points # 先保存
    RGB_INFO
    #points_raw = points_raw.cuda()

    #网络准备
    state_dict = torch.load(MODEL_PATH,map_location=ENV_MODEL)
    classifier = get_model(NUM_CLASSES)#.cuda()
    classifier.load_state_dict(state_dict['model_state_dict'])
    classifier.eval()
        
    #网络预测输出
    time_start = time()
    points = points.transpose(1, 0).contiguous()
    # points=torch.from_numpy(points).contiguous()
    # print(points)
    points = points.view(1, points.size()[0], points.size()[1])#.cuda()#reshape batchsize为1，第二维度为3 第三维度2500 
    pred,_ = classifier(points) #1*npoints*11矩阵
    # print(pred)   
    pred_choice = pred.data.max(2)[1]
    time_end = time()
    timecost = time_end - time_start   #运行所花时间
  
    print("网络预测时间:",timecost)
    #可视化
    #use_matplot(points_raw.cpu(),pred_choice.cpu(), 'pred_graph of sample {}'.format(index))    
    # print(points_raw)
    #返回
    return pred, pred_choice, points_raw,RGB_INFO #, pointslabel#, timecost


###############################################
#功能：显示预测的点云
#输入：包装好的点云
#输出：用于对点云进行标签采样的索引列表
############################################### 
# 返回pointnet++输出中的点云的labelled_index
def viz_pointcloud_o3d(labelled_points,display = False):#输入点云 输出点云显示
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
        # coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
        __pcd_vector.colors = o3d.utility.Vector3dVector(__colors)
        __pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])


        # target=o3d.utility.Vector3dVector(target)
        # target=target[:,0:3].reshape(-1,3)
        o3d.visualization.draw_geometries([__pcd_vector], window_name='2',
                                         point_show_normal=False,
                                         width=800,height=600,left = 500,top=300)
        return labelled_points

def data_align(__points_raw,__pred_choice):
    # __points_raw=torch.Tensor(__points_raw)
    __pred_choice = __pred_choice.view(-1,1) #N个一维数组变成 N*1的二维张量
    # __pred_choice=torch.Tensor(__pred_choice)
    # __pred_choice=__pred_choice.view(-1,1)
    __output = torch.cat((__points_raw,__pred_choice),1) # 拼接语义点云
    __output = __output.cpu().numpy()
    return __output

# ADD RGB INFO to the target point cloud
def ADD_RGB(__points_raw,__pred_choice,RGB_INFO):
    __pred_choice = __pred_choice.view(-1,1)
    __output=torch.cat((__points_raw,RGB_INFO,__pred_choice),1)
    __output = __output.cpu().numpy()
    return __output

# Save the pointcloud file to txt
def save_txt(filepath,pointcloud):
    with open (filepath,'w') as file:
        for point in pointcloud:
            x,y,z,R,G,B,pred=point
            file.write(f'{np.float32(x)} {np.float32(y)} {np.float32(z)} {R} {G} {B} {pred}\n')



# ADD RGB INFO to the target point cloud
def ADD_RGB1(__points_raw,RGB_INFO):
    # __pred_choice = __pred_choice.view(-1,1)
    __output=torch.cat((__points_raw,RGB_INFO),1)
    __output = __output.cpu().numpy()
    return __output

if __name__ == '__main__':
    

    #dataset path
    root = './dataset'
    datasetpath = 'WAIT_FOR_SAL'
    rootpath = os.path.join(root,datasetpath)
    
    vis = o3d.visualization.Visualizer()
    #创建播放窗口
    # vis.create_window()
    # pointcloud = o3d.geometry.PointCloud()
    # to_reset = True
    # vis.add_geometry(pointcloud)


    for index in range(1,60):
       
        print('第',index,'次实验开始')
        filepath='./sem-automatic-labeled'+str(index)+'.txt'

        _,pred_choice, points_raw,RGB_INFO= get_pred_pointcloud(rootpath,index,100000)
        
        
        stairs_points = data_align(points_raw,pred_choice)###预测结果再整合###

        # viz_pointcloud_o3d(stairs_points,display=True)
        # pointcloud=viz_pointcloud_o3d(stairs_points,display=True)


        # vis.run()

    
        

        target_points=ADD_RGB(points_raw,pred_choice,RGB_INFO)
        

    
# vis.destroy_window()
        # stairs_index = viz_pointcloud_o3d(stairs_points,windows_name="清晰楼梯识别",display=True)###预测结果显示+分割结果的标签index（原始数据中的）保存###