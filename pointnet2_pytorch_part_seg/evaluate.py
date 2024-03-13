###必须的模块###
import numpy as np
np.set_printoptions(threshold=np.inf)
#from collections import  Counter
import sys
sys.path.append("D:\Project_on_going\pointnet2_pytorch_part_seg")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path
###网络相关###
from models.pointnet2_sem_seg_msg import get_model 
import torch
import torch.nn.parallel
import torch.utils.data
###数据相关###
from data_utils.mydataset import MyDataSet
###显示相关###
import open3d as o3d
from Lib.visualizelib_seg_o3d import viz_pointcloud_o3d #不带标签显示 以通过测试
from Lib.visualizelib import *
###计时###
import time
import math

def count_files_in_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort(key=lambda x: int(os.path.basename(x)))

    result = {}
    for subfolder in subfolders:
        file_count = len(os.listdir(subfolder))
        result[os.path.basename(subfolder)] = file_count

    return result

###############################################
#功能：GT读取程序
#输入：
#输出：
#这里重复计算太多次
###############################################
def read_GT_data(root,index):

    #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
    datapath = {} #文件序数和文件完整路径对应字典
    dictNumber = 10 #总共有多少个文件夹
    fileslist = []
    
    file_counts = count_files_in_subfolders(root)
    for k,v in file_counts.items():
        fileslist.append(v)
    # fileslist = [22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
    t = 0
    tmp = []
    tmp.append(0)
    for k,v in file_counts.items():
        t += v
        tmp.append(t)
    # tmp = [0, 22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
    #字典赋值
    #法向量标注文件查找字典的赋值
    for dictOrder in range(dictNumber): #dictOrder 0,1,2,3,4,5,6,7,8,9
        #tmp = self.fileslist[dictOrder] 
        for filesOrder in range(0,fileslist[dictOrder]):#filesOrder 0到当前dictorder的长度
            datapath[filesOrder+tmp[dictOrder]] = os.path.join(root,str(dictOrder+1),'{}_{}_norm.txt'.format(dictOrder+1,filesOrder+1))
    #print(datapath)
    normals  = np.loadtxt(str(datapath[index]),usecols=(0,1,2)).astype(np.float32)
    return normals

###############################################
#功能：评估程序
#输入：
#输出：
###############################################
def evaluate_position_direction(_from_dataset_normal_direction,_from_ransac_direction):
    #TODO 欧式距离归一化作为评判标准；这个好不好呢？不一定
    # def euclid_dis(point1,point2):
    #     return math.sqrt(point1*point1+point2*point2)
    normal1 = _from_dataset_normal_direction
    normal2 = _from_ransac_direction
    def direction_cosine(normal1,normal2):
        if(len(normal1)!=len(normal2)):
            print('error input,x and y is not in the same space')
            return
        result1=0.0
        result2=0.0
        result3=0.0
        for i in range(len(normal1)):
            result1+=normal1[i]*normal2[i]   #sum(X*Y)
            result2+=normal1[i]**2     #sum(X*X)
            result3+=normal2[i]**2     #sum(Y*Y)
        return result1/(math.sqrt(result2)*math.sqrt(result3))
    return direction_cosine(normal1=normal1,normal2=normal2)

def evaluate_centrol_method():
    #TODO FPS的centrol和PCD的centrol之间的比较；
    #由于单纯是算法的比较，无GT，所以把FPS求中心点的输入分为2个，一个给PCD，一个给FPS
    pass
def evaluate_finishing():
    #TODO 球采样之后进行补全,补全成功占全部的比例；
    #TODO 可分为完整输入经过球采样补全和系统中的补全度；
    def finishing_ratio():
        pass
    pass
def evaluate_speed():
    #实际调用的速度；
    pass



def find_error_points(points,pred_choice,points_label):
   
    app = gui.Application.instance
    app.initialize()
    
    #测试输入
    #points = make_point_cloud(100, (0, 0, 0), 1.0)

    #输入不是pcd点云 所以先变成pcd点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    colors = []
    for i in range(len(pred_choice)):
        if pred_choice[i] != points_label[i]:
            colors.append([1,0,0])
        if pred_choice[i] == points_label[i]:
            if pred_choice[i] == 10:
                colors.append([0,0,1])
            else:
                colors.append([0,1,0])
    
    print("红的是预测错误的楼梯或非楼梯（FN 和 FP），绿的是预测准确的楼梯（TP），蓝的是预测正确的非楼梯（TN）")
    pcd.colors = o3d.utility.Vector3dVector(np.reshape(colors,(-1,3)))  #输入点云类别*3的矩阵
    
    #可视化
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("PointsCLoud", pcd)
 
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

if __name__ == '__main__':
    root = r'D:\Project_on_going\StairsSet\SZTUstairs\dataset_withlabell_eval_downsampled_stepnormal'
    read_GT_data(root,1)



 

