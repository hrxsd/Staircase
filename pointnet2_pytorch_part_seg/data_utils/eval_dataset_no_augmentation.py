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

class EvalDataSet(data.Dataset):
    def __init__(self, root, npoints):
        
        #导入外参
        self.npoints = npoints
        
        #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
        self.datapath = {} #文件序数和文件完整路径对应字典
        self.datapath_norms = {}
        self.root = root
        self.dictNumber = 5
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
        print("datapath:",str(self.datapath[index]))
        #Pointnet++的随机降采样算法
        try:
            choice = np.random.choice(len(pointLabel), self.npoints, replace=True)# 重新采样到self.npoints个点
            choice = np.asarray(choice)
            pointSet   = pointSet[choice, :] #resample
            pointLabel = pointLabel[choice]
        except:
            pass
        
        # 直通滤波
        remain_index_list = []
        for i in range(len(pointLabel)):
            if pointSet[i][0] <= 1 and pointSet[i][1] <= 1 and pointSet[i][2] <= 1.2:
                remain_index_list.append(i)
        pointSet = pointSet[remain_index_list][:]
        pointLabel = pointLabel[remain_index_list]    
    
         # 去中心化 # 归一化#计算到原点的最远距离
        center = np.expand_dims(np.mean(pointSet, axis = 0), 0)
        pointSet = pointSet - center # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale

        pointSet   = torch.from_numpy(pointSet)
        pointLabel = torch.from_numpy(pointLabel)

        return pointSet,pointLabel,center,dist

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    root = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled'
    npoints = 2048
    dataset = MyDataSet(root=root,npoints=npoints)
    data = dataset.__getitem__(1)
    print(data)
