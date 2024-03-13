###必须的模块###
import numpy as np
from time import time
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

import os
import torch
from time import time


class PointCloudPredictor:
    def __init__(self, model_root, model_version, num_classes, device='cuda'):

        self.model_root = model_root
        self.model_version = model_version
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model_final_path = r'checkpoints/best_model.pth'
        self.model_path = os.path.join(self.model_root, self.model_version, self.model_final_path)
        self.classifier = self.load_classifier()

    def load_classifier(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        classifier = get_model(self.num_classes)
        classifier.load_state_dict(state_dict['model_state_dict'])
        classifier.eval()
        return classifier

    def get_pred_pointcloud(self, points):
        # points = points_set
        points_raw = points  # 先保存

        time_start = time()
        points = points.transpose(1, 0).contiguous()
        points = points.view(1, points.size()[0], points.size()[1])
        pred, _ = self.classifier(points)
        pred_choice = pred.data.max(2)[1]
        time_end = time()
        timecost = time_end - time_start
        # print('prediction time:',timecost)

        return pred, pred_choice, points_raw


# def get_pred_pointcloud(points_set):

#     #外参
#     MODEL_ROOT = './log/sem_seg/experiment'
#     MODEL_VERSION = '7'

#     MODEL_FINALPATH = r'checkpoints/best_model.pth'
#     MODEL_PATH = os.path.join(MODEL_ROOT,MODEL_VERSION,MODEL_FINALPATH)
#     ENV_MODEL = torch.device('cpu')
#     NUM_CLASSES = 11

#     #数据提取

#     points=points_set
#     points_raw = points # 先保存


#     #网络准备
#     state_dict = torch.load(MODEL_PATH,map_location=ENV_MODEL)
#     classifier = get_model(NUM_CLASSES)#.cuda()
#     classifier.load_state_dict(state_dict['model_state_dict'])
#     classifier.eval()

#     #网络预测输出
#     time_start = time()
#     points = points.transpose(1, 0).contiguous()
#     # points=torch.from_numpy(points).contiguous()
#     # print(points)
#     points = points.view(1, points.size()[0], points.size()[1])#.cuda()#reshape batchsize为1，第二维度为3 第三维度2500
#     pred,_ = classifier(points) #1*npoints*11矩阵
#     # print(pred)
#     pred_choice = pred.data.max(2)[1]
#     time_end = time()
#     timecost = time_end - time_start   #运行所花时间

#     print("网络预测时间:",timecost)
#     #可视化
#     #use_matplot(points_raw.cpu(),pred_choice.cpu(), 'pred_graph of sample {}'.format(index))
#     # print(points_raw)
#     #返回
#     return pred, pred_choice, points_raw #, pointslabel#, timecost


###############################################
#功能：显示预测的点云
#输入：包装好的点云
#输出：用于对点云进行标签采样的索引列表
###############################################
# 返回pointnet++输出中的点云的labelled_index
def viz_pointcloud_o3d(labelled_points):  # 输入点云 输出点云显示
    __pcd_vector = o3d.geometry.PointCloud()  # 生成Pointcloud对象，o3d需要
    __colors = np.zeros((len(labelled_points), 3))  # 颜色矩阵
    labelled_index = []  # 保存标签的index
    for j in range(len(labelled_points)):  # 颜色根据label变化
        if labelled_points[j][3] == 11:
            __colors[j] = [0,0,1]
            labelled_index.append(j)
        elif labelled_points[j][3] != 10:
            __colors[j] = [1, 0, 0]  # RGB 确定为红色
            labelled_index.append(j)
        else:
              __colors[j] = [0,0,0] # RGB确定为蓝色

    __pcd_vector.colors = o3d.utility.Vector3dVector(__colors)
    __pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])
    return __pcd_vector    
    
    
    
    


    return labelled_index#,__pcd_vector

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

def custom_input(prompt=""):
    readline.set_startup_hook(lambda:readline.insert_text(""))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


###############################################
# 功能：点云拟合平面方程
# 输入：点云
# 输出：__pcd_for_ransac_return:只包含inlier points
# 系统超参数：ransac_n：如果一个聚类结果中点的数量小于ransac_n会抛出异常，用以防止该情况。
#           即使点云数量大于该参数，由于拟合会进行降采样（只采用inlier）所以输出的点依然会小于ransace_n
#           这会导致sampling_fps_points跳过。
###############################################
def get_ransac_pointcloud(__pcd, __labels, __class_number, display=False):  # class_number 是聚类结果数量

    # pcd经过采样的  数据的index 早就变了
    ###############################################
    # 功能：
    # 输入：
    # 输出：
    ###############################################
    def _get_group_point_cloud(__pcd2, __labels2, __class_number2):
        __point_list_idx = [[] for _ in range(__class_number2)]  # 存放不同类点的index
        # __pcd_list_idx为二维数组  第一维是分类，第二维是对应点的index
        for idx_point in range(0, len(__pcd2.points)):
            __point_list_idx[int(__labels2[idx_point])].append(idx_point)  # 对应点的label就是__pcd_list中存放的位置的index
        # print("pcdlistidx",__point_list_idx)

        # 由__point_list_idx导出__pcd_for_ransac
        # __pcd_for_ransac有2维，第一维分类，第二维是pcd对象(二维是点个个数，第三维是点的位置)
        __pcd_for_ransac = []  # 组成pcd文件
        for a in range(len(__point_list_idx)):  # classnumber 既是pcdlistindex的第一维
            # 在分类维度
            __pcd_for_ransac.append(o3d.geometry.PointCloud())
            __point_tmp = []
            for b in range(len(__point_list_idx[a])):  # 在点云个数维度循环
                __point_tmp.append(__pcd.points[__point_list_idx[a][b]])

            __pcd_for_ransac[a].points = o3d.utility.Vector3dVector(__point_tmp)
        for i in range(__class_number2):
            print("pcd_for_ransac_{}".format(i), len(__pcd_for_ransac[i].points))
        return __pcd_for_ransac

    # __pcd_for_ransac有2维，第一维分类，第二维是pcd对象(也可以说二维是点个个数，第三维是点的位置)
    __pcd_for_ransac = _get_group_point_cloud(__pcd, __labels, __class_number)
    __pcd_for_ransac_return = [[] for _ in range(__class_number)]  # 用来装o3d.geometry.PointCloud()
    ransac_n = 20
    plane_params = [[] for _ in range(__class_number)]
    for idx_class in range(0, __class_number):
        if len(__pcd_for_ransac[idx_class].points) < ransac_n:  # 如果一个聚类结果中点的数量小于ransac_n会抛出异常，用以防止该情况
            print("the points in class:%d r below than ransan_c's request" % (idx_class))
            continue
        plane_model, inliers = __pcd_for_ransac[idx_class].segment_plane(distance_threshold=0.002,  # 内点到平面模型的最大距离
                                                                         ransac_n=ransac_n,  # 用于拟合平面的采样点数
                                                                         num_iterations=100)  # 最大迭代次数
        [a, b, c, d] = plane_model
        plane_params[idx_class] = [a, b, c, d]
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        # 平面内点点云
        inlier_cloud = __pcd_for_ransac[idx_class].select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        # 平面外点点云
        outlier_cloud = __pcd_for_ransac[idx_class].select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 0, 1.0])
        # 可视化平面分割结果
        if display == True:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud] + [coord],
                                              window_name="拟合平面{}".format(idx_class),
                                              width=800, height=600, left=1500, top=500)
        __pcd_for_ransac_return[idx_class] = inlier_cloud
    return __pcd_for_ransac_return, plane_params


###############################################
# 功能：从点云中三维重建出拟合的平面并且返回平面中心点
# 输入：pcdlist pcdraw
# 输出：一个列表：_pcd_meanpoints_fragments:[_pcd_mean_point,_pcd_list[_pcd_idx],_mesh]
# 系统超参数：sampling_fps_point 使用FPS进行多边形的中心点计算，不足fps最小要求时会跳过。
# TODO 不足时采用传统平均方法
###############################################
def get_3dreconstruct_and_meanpoints(_pcd_list, _pcd_raw, plane_params, sampling_fps_point=20,  ##FPS需要的最低点的个数
                                     need_steps_fragments_correction=True):
    _pcd_meanpoints_fragments = [[] for _ in range(len(_pcd_list))]  # 返回存放全部的中心点和楼梯碎片用以碎片补偿
    _fps_ndarray = []  # 存放某一循环的中心点的位置
    _pcd_list_len = len(_pcd_list)
    for _pcd_idx in range(_pcd_list_len):

        # 判断长度符合要求与否  这个也是一种降采样
        try:
            if len(_pcd_list[_pcd_idx].points) < sampling_fps_point:
                print("the points in this plane cannot reached the least points requset of FPS")
                continue
        except AttributeError:
            continue
        # 中心点
        # _fps_pcd = _pcd_list[_pcd_idx].farthest_point_down_sample(sampling_fps_point)
        _fps_ndarray = np.mean(np.asarray(_pcd_list[_pcd_idx].points), axis=0)  # 返回1*3
        print("centrol point:", _fps_ndarray)

        # 存储单个中心点用
        _pcd_mean_point = o3d.geometry.PointCloud()
        _pcd_mean_point.points = o3d.utility.Vector3dVector(_fps_ndarray.reshape(1, 3))
        _pcd_mean_point.paint_uniform_color([0, 0, 1])

        # 重建
        _mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(_pcd_list[_pcd_idx], alpha=2)

        # 可视化
        # coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
        # o3d.visualization.draw_geometries([_pcd_mean_point,_pcd_raw,_mesh]+[coord],
        #                                     window_name="三维重建和中心点{}".format(_pcd_idx),
        #                                     point_show_normal=False,
        #                                     width=800,height=600,left =1500,top=500,
        #                                     mesh_show_wireframe=False,mesh_show_back_face=True)

        # True的话函数会构建返回值
        if need_steps_fragments_correction == True:
            _pcd_meanpoints_fragments[_pcd_idx] = [_pcd_mean_point, _pcd_list[_pcd_idx], _mesh, plane_params[_pcd_idx]]

    # print("pcd_meanpoints_fragments", len(_pcd_meanpoints_fragments))

    # True的话函数返回有值
    if need_steps_fragments_correction == True:
        return _pcd_meanpoints_fragments

###############################################
# 功能：从中心点求解出楼梯高度和宽度，另外显示面法向量
# 输入：重建之后得出的中心点和 ransac得出的法向量
# 输出：实际的距离（以米为单位）
###############################################
def modelling_stairs_params(pcd_list, depthscale=2500):
    # 先判断非空连续集合
    tmp_list = []
    for i in range(len(pcd_list)):
        if len(pcd_list[i]) != 0:
            tmp_list.append(i)

    # 在plane 只有一个的情况下 引发了 Nonetype 异常，因为后面要用。
    if (len(tmp_list)) <= 1:
        return tmp_list

    # 相近idx之间的距离
    tmp_list_for_idx_distance = []
    for j in range(len(tmp_list) - 1):
        tmp_list_for_idx_distance.append(tmp_list[j + 1] - tmp_list[j])

    stairs_params = [[], []]  # 第一个是height，第二个是depth
    stairs = []

    def numpy_flat(a):
        return list(np.array(a).flat)

    # for idx in range(tmp_list):#N个台阶算n-1次
    #     previous = pcd_list[idx][0]
    #     next = pcd_list[idx+1][0]
    #     pre_y  = numpy_flat(np.asarray(previous.points))[1]
    #     pre_z  = numpy_flat(np.asarray(previous.points))[2]
    #     next_y = numpy_flat(np.asarray(next.points))[1]
    #     next_z = numpy_flat(np.asarray(next.points))[2]
    #     stairs_params.append(abs(pre_y - next_y)*depthscale/1000)
    #     stairs_params.append(abs(pre_z - next_z)*depthscale/1000)

    # height
    first_step_centrol = pcd_list[tmp_list[0]][0]
    second_step_centrol = pcd_list[tmp_list[1]][0]
    previous = first_step_centrol
    next = second_step_centrol
    pre_y = numpy_flat(np.asarray(previous.points))[1]
    pre_z = numpy_flat(np.asarray(previous.points))[2]
    next_y = numpy_flat(np.asarray(next.points))[1]
    next_z = numpy_flat(np.asarray(next.points))[2]

    stairs_params[0] = (abs(pre_y - next_y) * depthscale / 1000) / tmp_list_for_idx_distance[0]
    # depth
    stairs_params[1] = (abs(pre_z - next_z) * depthscale / 1000) / tmp_list_for_idx_distance[0]
    print("stairs_params:", stairs_params)
    print("tmp_list", tmp_list)
    return tmp_list


if __name__ == '__main__':
    # 创建预测器对象
    model_root = './log/sem_seg/experiment'
    model_version = '7'
    num_classes = 11
    predictor = PointCloudPredictor(model_root, model_version, num_classes)

    # 假设有一个点云数据集 points_set
    # 进行预测
    # pred, pred_choice, points_raw, timecost = predictor.get_pred_pointcloud(points_set)

    # 打印预测结果和时间花费