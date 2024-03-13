###必须的模块###
import numpy as np
from time import time
np.set_printoptions(threshold=np.inf)
from collections import  Counter
import sys
sys.path.append("D:\\Project_on_going\\pointnet2_pytorch_part_seg")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path

###网络相关###
from models.pointnet2_sem_seg_msg import get_model 
import torch
import torch.nn.parallel
import torch.utils.data

###数据相关###
from data_utils.eval_dataset_no_augmentation import EvalDataSet

###显示相关###
import open3d as o3d
from data_operate_kit.visualizelib import *

###最远点采样###
from models.pointnet2_utils import *

###评估#####
from robot_math import *
import math
from experiment_code.evaluate_lib import *

###DataRecord###
import datetime
from data_operate_kit.data_recorder import *

###plane finder###
from experiment_code.plane_finder import *

###modelling###
from experiment_code.modelling import *

#功能：从原始点云当中提取出识别和分割的成像点云
#输入：raw point cloud 经由dataloader 送出
#输出：pred, pred_choice, points_raw
#注意：改成了CPU版本
def get_pred_pointcloud(model_path,model_version,dataset_path,index,npoints):

    #外参
    MODEL_ROOT = model_path 
    MODEL_VERSION = model_version
    
    MODEL_FINALPATH = r'checkpoints/model.pth'
    MODEL_PATH = os.path.join(MODEL_ROOT,MODEL_VERSION,MODEL_FINALPATH)
    ENV_MODEL = torch.device('cpu')
    NUM_CLASSES = 12

    #数据提取
    dataset = EvalDataSet(root=dataset_path,npoints = npoints)
    points,pointslabel,center,scale_rate = dataset.__getitem__(index) #npoints*3+npoints*1
    points_raw = points # 先保存
    points_raw = points_raw.cuda()

    #网络准备
    state_dict = torch.load(MODEL_PATH,map_location=ENV_MODEL)
    classifier = get_model(NUM_CLASSES).cuda()
    classifier.load_state_dict(state_dict['model_state_dict'])
    classifier.eval()
        
    #网络预测输出
    #time_start = time.time()
    points = points.transpose(1, 0).contiguous()#3*2500
    points = points.view(1, points.size()[0], points.size()[1]).cuda()#reshape batchsize为1，第二维度为3 第三维度2500 
    pred,_ = classifier(points) #1*npoints*11矩阵   
    pred_choice = pred.data.max(2)[1]
    #time_end = time.time()
    #timecost = time_end - time_start   #运行所花时间
        
    #可视化
    #use_matplot(points_raw.cpu(),pred_choice.cpu(), 'pred_graph of sample {}'.format(index))    
        
    #返回
    return pred, pred_choice, points_raw, pointslabel,MODEL_VERSION,center,scale_rate#, timecost

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
                                        width=800,height=600,left = 750,top=250)  
    return labelled_index 

def data_align(__points_raw,__pred_choice):
    __pred_choice = __pred_choice.view(-1,1) #N个一维数组变成 N*1的二维张量
    __output = torch.cat((__points_raw,__pred_choice),1) # 拼接语义点云
    __output = __output.cpu().numpy()
    return __output

def viz_pointcloud_distribution(labelled_points):#输入点云 输出点云显示
    __pcd_vector = o3d.geometry.PointCloud() #生成Pointcloud对象，o3d需要
    __colors = np.zeros((len(labelled_points),3)) # 颜色矩阵
    labelled_index = []#保存标签的index
    for j in range(len(labelled_points)): #颜色根据label变化
        if labelled_points[j][3] == 10:
            __colors[j] = [0,0,0]
        elif labelled_points[j][3] == 11:
            __colors[j] = [0,0,1]
        elif labelled_points[j][3] % 2 == 0:
            __colors[j] = [1,0,0] # RGB 确定为红色
            labelled_index.append(j)
        else:
            __colors[j] = [0,1,0] # RGB确定为蓝色

    __pcd_vector.colors = o3d.utility.Vector3dVector(__colors)
    __pcd_vector.points = o3d.utility.Vector3dVector(labelled_points[:,:3])
    return __pcd_vector

# illustrate eigen vector
def display_vector(start_points,direction):
    points = [start_points,direction]
    points = o3d.utility.Vector3dVector(points)
    lines = [[0,1]]
    lines=o3d.utility.Vector2iVector(lines)
    vector = o3d.geometry.LineSet(points,lines)
    vector.colors = o3d.utility.Vector3dVector([[1,0,0],[1,0,0]])
    return vector

if __name__ == '__main__':

    #data recorder init
    experiment = 'stairs_params'
    data_record_path = 'D:\\Project_on_going\\pointnet2_pytorch_part_seg\\experiment_data\\20230910-generalization'
    dr = data_recorder(experiment_name=experiment,write_path=data_record_path,files_order=False)

    #model init
    model_path = 'D:\\Project_on_going\\pointnet2_pytorch_part_seg\\log\\sem_seg\\experiment\\ablationstudy_20230908'
    #model_path = 'E:\\Trianed_Model'
    model_version = 'None'

    #pred dataset path
    root = 'D:\\Project_on_going\\generalization'
    datasetpath = 'downsampled'
    rootpath = os.path.join(root,datasetpath)

    #warning for debug
    warning_index = []
    
    #network load
    for index in range(0,50):
        print('第',index,'次实验开始')
        current_datetime = datetime.datetime.now()
        current_date = current_datetime.date()

        # stairs detection
        _, pred_choice, points_raw, gt_labels,MODEL_VERSION,center,scale_rate = get_pred_pointcloud(model_path,model_version,rootpath,index,5000)
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(points_raw[:,:3].cpu()))
        stairs_points = data_align(points_raw,pred_choice)###预测结果再整合###
        stairs_index = viz_pointcloud_o3d(stairs_points,windows_name="清晰楼梯识别",display=False)###预测结果显示+分割结果的标签index（原始数据中的）保存###
        stairs_points = stairs_points[stairs_index]###重新采样###
        stairs_points_distribution = stairs_points
        stairs_pcd = o3d.geometry.PointCloud()
        stairs_pcd.points = o3d.utility.Vector3dVector(stairs_points[:,:3]) #with no labelled 

        # plane finder
        pcd_plane_fit,plane_params,pcd_stairs_list = get_ransac_pointcloud(stairs_pcd,list(stairs_points[:,3]),display=False)
        
        # central point and modelling
        pcd_meanpoints_planeparams_points = get_meanpoints(pcd_plane_fit,pcd_stairs_list,plane_params)
        try:
            idx_list,stairs_params,rotate_matrix_h,rotate_matrix_d,pre_correct,next_correct,pre,next,_rotated_points,targetpoint,startpoint = modelling_stairs_params(pcd_meanpoints_planeparams_points,center,scale_rate)
        except ValueError: # FPS not passed
            print("ValueError")
            continue
        if len(idx_list) <= 1: #这时不适合判断
            print("Idx list length less than or equal to 1")
            continue

        # evaluate
        try:
            dr.record_data('date',current_date)   
            dr.record_data('dataset',datasetpath)
            dr.record_data('index',index)
            dr.record_data('version',MODEL_VERSION)

            # step's params
            # pred
            pred_step_params = stairs_params
            dr.record_data('predited params',pred_step_params)
            print('predited params:',pred_step_params)
            # GT
            gt_step_params_path =r'D:\\Project_on_going\\generalization\\groundtruthparams'
            gt_step_params = read_GT_stairs_params_data(gt_step_params_path,index)/100.0
            print("measured_gt_step_params",gt_step_params)
            dr.record_data('GT params',gt_step_params)
            # compute
            height_error =  abs(gt_step_params[0] - pred_step_params[0])
            depth_error = abs(gt_step_params[1] - pred_step_params[1])
            dr.record_data('height error',height_error)
            dr.record_data('depth error',depth_error)
            pred_choice = set(list(pred_choice.cpu().numpy()[0]))
            gt_labels = set(list(gt_labels.cpu().numpy()))
            dr.record_data('step num',len(pred_choice)-1)
            dr.record_data('GT step num',len(gt_labels)-1)

            # centrol point
            # pred
            pred_step_centrol_point = pcd_meanpoints_planeparams_points[idx_list[0]][0][:3]
            dr.record_data('predited central point',pred_step_centrol_point)
            # GT
            gt_centrol_point_path =r'D:\\Project_on_going\\generalization\\downsampled_centralpoints'
            gt_centrol_point = read_GT_centrol_data(gt_centrol_point_path,index,center=center,scale_rate=scale_rate)
            #print("gt_centrol_point",gt_centrol_point)
            #print("the chosen centrol_point_gt index",idx_list[0])
            #print("the chosen centrol_point_gt",gt_centrol_point[idx_list[0]])
            gt_step_centrol_point = gt_centrol_point[idx_list[0]]
            dr.record_data('GT central point',gt_step_centrol_point)
            central_point_error = evaluate_centrol_distance(gt_step_centrol_point,pred_step_centrol_point)
            dr.record_data('centrol point error',central_point_error)
            #print("central point error:",central_point_error)

            # compare gt central point derived params and gt params
            # gt central point
            # prev = rotate_matrix_d @ rotate_matrix_h @ gt_centrol_point[0]
            # next = rotate_matrix_d @ rotate_matrix_h @ gt_centrol_point[1]
            # # compute the derived params
            # derived_stairs = abs(next-prev)
            # print("derived_gt_step_params:",derived_stairs*2500/1000)

            # normal 评估
            dr.record_data('preditec normal',plane_params[0][:3])
            eval_norm_root =r'D:\\Project_on_going\\generalization\\downsampled_stepnormal'
            normal_gt = read_GT_data(eval_norm_root,index)
            dr.record_data('GT normal',normal_gt[0])
            error_norm = evaluate_position_direction(normal_gt[0],plane_params[0][:3])
            dr.record_data('normal error',error_norm)

            # pcd_1 = o3d.geometry.PointCloud()
            # pcd_1.points = o3d.utility.Vector3dVector(np.asarray(gt_step_centrol_point).reshape(1,3))
            # pcd_1.paint_uniform_color([0, 0, 1])
            # pcd_2 = o3d.geometry.PointCloud()
            # pcd_2.points = o3d.utility.Vector3dVector(np.asarray(pred_step_centrol_point).reshape(1,3))
            # pcd_2.paint_uniform_color([0, 0, 1])

            coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            vector = display_vector(startpoint,targetpoint)
            
            pcd_distribution = viz_pointcloud_distribution(stairs_points_distribution)
              
            # pcd_rotated_points = o3d.geometry.PointCloud()
            # pcd_rotated_points.points = o3d.utility.Vector3dVector(np.asarray(_rotated_points).reshape(-1,3))
            # pcd_rotated_points.paint_uniform_color([0, 0, 0])

            # pcd_pre = o3d.geometry.PointCloud()
            # pcd_pre.points = o3d.utility.Vector3dVector(np.asarray(pre).reshape(1,3))
            # pcd_pre.paint_uniform_color([0, 0, 1])
            # pcd_next = o3d.geometry.PointCloud()
            # pcd_next.points = o3d.utility.Vector3dVector(np.asarray(next).reshape(1,3))
            # pcd_next.paint_uniform_color([0, 0, 1])
            # pcd_cp_gt = o3d.geometry.PointCloud()
            # pcd_cp_gt.points = o3d.utility.Vector3dVector(np.asarray(gt_step_centrol_point).reshape(1,3))
            # pcd_cp_gt.paint_uniform_color([0, 0, 1])
            # pcd_3 = o3d.geometry.PointCloud()
            # pcd_3.points = o3d.utility.Vector3dVector(np.asarray(stairs_pcd.points).reshape(-1,3))
            # pcd_3.paint_uniform_color([1, 0, 0])   

            rotated_point = np.asarray(stairs_pcd.points).transpose(1,0)
            rotated_point =  rotate_matrix_d @ rotate_matrix_h @ rotated_point
            pcd_r = o3d.geometry.PointCloud()
            pcd_r.points = o3d.utility.Vector3dVector(rotated_point.transpose(1,0))
            pcd_r.paint_uniform_color([0, 1, 0])    

            cp_gt_correct = []
            for i in range(len(gt_centrol_point)):
                cp_gt_correct.append( rotate_matrix_d @ rotate_matrix_h @ gt_centrol_point[i][:3])#.transpose(1,0)#gt_step_centrol_point
            pcd_cp_gt_correct = o3d.geometry.PointCloud()
            pcd_cp_gt_correct.points = o3d.utility.Vector3dVector(np.asarray(cp_gt_correct).reshape(-1,3))
            pcd_cp_gt_correct.paint_uniform_color([0, 0, 1])

            pcd_pre_correct = o3d.geometry.PointCloud()
            pcd_pre_correct.points = o3d.utility.Vector3dVector(pre_correct.reshape(-1,3))
            pcd_pre_correct.paint_uniform_color([0, 0, 0])
            pcd_next_correct = o3d.geometry.PointCloud()
            pcd_next_correct.points = o3d.utility.Vector3dVector(next_correct.reshape(-1,3))
            pcd_next_correct.paint_uniform_color([0, 0, 0])   

            # o3d.visualization.draw_geometries([coord]+#[pcd_distribution]+
            #                                   [pcd_pre_correct]+
            #                                   [pcd_next_correct]+
            #                                   [vector]+
            #                                   [pcd_r]+
            #                                   [pcd_cp_gt_correct], window_name='{}'.format(index),
            #                                   point_show_normal=False,
            #                                   width=800,height=600,left = 600,top=250)  

            if (len(pcd_plane_fit)-1) != len(gt_centrol_point):
                dr.record_data('note',"predited step num not equals to the gt")
                print('predited step num not equals to the gt:',index)
            else:
                dr.record_data('note',"no prob")
        except IndexError:
            print("WARNING")
            warning_index.append(index)
            raise Exception
        dr.record2excel()   
    print("warning_index:",warning_index)
    dr.record2excel()   
