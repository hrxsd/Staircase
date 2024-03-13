import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
sys.path.append("D:\pytorch_project\pointnet2_pytorch-master")

from data_utils.mydataset import MyDataSet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os.path
import json
from models.pointnet2_sem_seg_msg import get_model 
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable



def use_matplot(points_list,labels_list,title):
    x = points_list[:, 0]  # x position of point
    y = points_list[:, 1]  # y position of point
    z = points_list[:, 2]  # z position of point
    fig = plt.figure(figsize=(10,10))
    plt.axis('off')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,c=labels_list,cmap='bwr',s= 10,marker=".")
    plt.title("{}".format(title))
    plt.show()

def viz_raw_pointcloud(rootpath,index):
    dataset = MyDataSet(root=rootpath,label12=2,npoints=2500)
    points,labels = dataset.__getitem__(index)
    use_matplot(points,labels,'raw_graph of sample {}'.format(index))


def viz_seg_pointcloud(rootpath,index):
    #数据提取
    dataset = MyDataSet(root=rootpath,label12=2,npoints=2500)
    points,_ = dataset.__getitem__(index) #2500*3+2500*1
    points_raw = points # 先保存
    
    #网络准备
    state_dict = torch.load('D:\pytorch_project\pointnet2_pytorch-master\log\sem_seg\segmsg_newdataset_nopointnormalize\\checkpoints\\best_model.pth')
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
    use_matplot(points_raw.cpu(),pred_choice.cpu(), 'pred_graph of sample {}'.format(index))

if __name__ == '__main__':
    rootpath = 'D:\pytorch_project\StairsSet\downsampled'
    for i in range(70):
        filesorder = i
        #viz_raw_pointcloud(rootpath,filesorder)
        viz_seg_pointcloud(rootpath,filesorder)

# 7 35 36 45失败
# 37 这种复杂的也学到了

