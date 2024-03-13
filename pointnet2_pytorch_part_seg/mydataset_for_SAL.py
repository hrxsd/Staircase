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

class MyDataSet(data.Dataset):
    def __init__(self, root,npoints):
        
        #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
        self.npoints = npoints
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
        pointSet     = np.loadtxt(str(self.datapath[index]),usecols=(0,1,2)).astype(np.float32)
        RGB_INFO     = np.loadtxt(str(self.datapath[index]),usecols=(3,4,5)).astype(np.float32)# 读取点云
        # pointLabel   = np.loadtxt(str(self.datapath[index]),usecols=(6)).astype(np.int64)           # 读取标签1
        pointNormal  = None
        # print(RGB_INFO)
        # try:
        #     # 可能引发异常的代码块
        #     pointNormal = np.loadtxt(str(self.datapath[index]),usecols=(7,8,9)).astype(np.float32)
        # except IndexError:
        #     # 异常处理代码块
        #     pass

        # # 检查数据合规范要求 
        # for i in np.nditer(pointLabel):
        #     if pointLabel[i] >10 or pointLabel[i]<0:
        #         print("datapath:",self.datapath[index])
        #         print("row{}",format(i))
        #         raise Exception("标记数据不合规")
        
        #  #Pointnet++的随机降采样算法
        #  try:
        #     choice = np.random.choice(np.size(pointSet,0), self.npoints, replace=True)# 重新采样到self.npoints个点
        #     choice = np.asarray(choice)
        #     pointSet   = pointSet[choice, :] #resample
        #     # pointLabel = pointLabel[choice]
        #     # try:
        #     #     # pointNormal = pointNormal[choice,:]
        #     # except NameError:
        #     #     pass
        # except:
        #     pass

    
         # 去中心化 # 归一化#计算到原点的最远距离
        pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale

        pointSet   = torch.from_numpy(pointSet)
        RGB_INFO=torch.from_numpy(RGB_INFO)
        # pointLabel = torch.from_numpy(pointLabel)
        # try:
        #     pointNormal = torch.from_numpy(pointNormal)
        # except NameError:
        #     pass
        # return pointSet,pointLabel
        # print(np.size(pointSet.cpu()))

        return pointSet,RGB_INFO

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    root = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled'
    npoints = 2048
    dataset = MyDataSet(root=root,npoints=npoints)
    data = dataset.__getitem__(1)
    print(data)
