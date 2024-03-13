import numpy
import rospy

from geometry_msgs.msg import PointStamped, Vector3Stamped
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
###必须的模块###
import numpy as np
np.set_printoptions(threshold=np.inf)
from collections import  Counter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os.path

import open3d as o3d

###网络相关###
from models.pointnet2_sem_seg_msg import get_model
import torch
import torch.nn.parallel
import torch.utils.data

from data_utils.data_preprocess import compute_steps_num
###显示相关###
from Lib.visualizelib import *
###最远点采样###
from models.pointnet2_utils import *
import readline
##网络模型##
import sem_autuomatic
import threading
import time
import math
import plane_finder
import  modelling
from robot_math import *
import test1


global d
d = None

Arrow = None
# 回调获取点云话题数据
def callback_pcl(data):
    global d
    d = data


# 转换点云格式
def Point_cloud():
    global d
   
    ss = point_cloud2.read_points(d, field_names=("x", "y", "z"), skip_nans=True)
    ss = list(ss)
    points = np.array(ss)
    return points

def get_pointset(pcd_points):
    point_cloud_array = np.asarray(pcd_points,dtype=np.float32)
    xyz_coordinates = point_cloud_array[:, :3]
    return xyz_coordinates

##To downsample the pointcloud
def random_downsample(pointSet,npoints):
    try:
        # choice = np.random.choice(np.size(pointSet,0), npoints, replace=True)# 重新采样到self.npoints个点
        # choice = np.asarray(choice)
        # pointSet   = pointSet[choice, :] #resample
        index1=pointSet.shape[0]
        choice=[]
        value=int(index1/2500)
        for i in  range(2500):
            choice.append(i*value)
        choice=np.asarray(choice)
        pointSet=pointSet[choice,:]

    except:
        pass

    # 直通滤波
    remain_index_array = np.array([],dtype=int)
    for i in range(len(pointSet)):
        if pointSet[i][0] <= 1 and pointSet[i][1] <= 1 and pointSet[i][2] <= 1.2:
            remain_index_array=np.append(remain_index_array,i)
    pointSet = pointSet[list(remain_index_array)][:]

    center = np.expand_dims(np.mean(pointSet, axis=0), 0)
    # pointSet = pointSet - center  # center
    dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis=1)), 0)
    # pointSet = pointSet / dist  # scale

    #pointSet = pointSet.astype(np.float32)
    pointSet = torch.from_numpy(pointSet.astype(np.float32))
    return pointSet,center,dist

def update_visualization(reult):
    to_reset = True
    # 沿着x轴旋转
    angle = np.radians(180)
    reult.rotate(np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]), center=(0, 0, 0))


    vis.add_geometry(reult)

    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    vis.poll_events()
    vis.update_renderer()

def compute_steps_num(label1,not_step):
    try:
        step_idx_list = []
        for i in range(len(label1)):
            if label1[i] != not_step:
                step_idx_list.append(i)
        _set_for_count = set(label1)
        if not_step in _set_for_count:
            _set_for_count.remove(not_step)

        print(int(max(_set_for_count)))
        return int(max(_set_for_count))

    except  ValueError as e:
        return -1

# def draw_normal(normal,start_point):
#     # 创建一个几何图元表示箭头
#     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02,cone_radius =0.04,cylinder_height=0.2,
#                                                cone_height=0.1,
#                                                resolution=30,
#                                                cylinder_split=4,
#                                                cone_split=1)
#     arrow.paint_uniform_color([0, 0, 1])

    # 计算箭头的旋转矩阵，使箭头方向与法向量一致
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0],np.array(normal,dtype=np.float64))  # 根据需要进行调整

    # 对箭头进行旋转和平移
    # arrow.transform(np.eye(4))
    
    # arrow_direction_normalized = normal / np.linalg.norm(normal)
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0], arrow_direction_normalized)

    # arrow.rotate(rotation_matrix,center=[0,0,0])
    # arrow.translate(start_point)
    return arrow

def rotate_point_cloud(points, angle, axis):
    """
    旋转点云

    参数：
    points: numpy数组,形状为(N, 3),表示点云中的N个点的XYZ坐标
    angle: float,旋转角度（以弧度为单位）
    axis: numpy数组,形状为(3,),表示旋转轴的XYZ分量

    返回值：
    numpy数组,形状为(N, 3),表示旋转后的点云坐标
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([[cos_theta + axis[0] ** 2 * (1 - cos_theta), 
                                 axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta, 
                                 axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
                                [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta, 
                                 cos_theta + axis[1] ** 2 * (1 - cos_theta), 
                                 axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
                                [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta, 
                                 axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta, 
                                 cos_theta + axis[2] ** 2 * (1 - cos_theta)]])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points




if __name__ == '__main__':
    rospy.init_node('points', anonymous=False)
    rospy.Subscriber('/front_camera/depth/color/points', PointCloud2, callback_pcl)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 显示坐标轴
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    # 初始化o3d格式点云
    points3d = o3d.geometry.PointCloud()
    stairs_pcd = o3d.geometry.PointCloud()
    # 设置点云大小，背景颜色
    render_option = vis.get_render_option()
    render_option.point_size = 5
    # render_option.background_color = np.asarray([0, 0, 0])
    # 加载点云到窗口里
    vis.add_geometry(stairs_pcd)

    # 用来发布中心点和法向量
    center_pub = rospy.Publisher('/stair_center', PointStamped, queue_size=10)
    # normal_pub = rospy.Publisher('/stair_normal', Vector3Stamped, queue_size=10)

    rate = rospy.Rate(30)
    # 网络的初始化
    # 创建预测器对象
    model_root = './log/sem_seg/experiment'
    model_version = '7'
    num_classes = 12
    # 记得改
    predictor =sem_autuomatic.PointCloudPredictor(model_root, model_version, num_classes)

    while not rospy.is_shutdown():
        # 订阅的话题是否有数据
        if not d is None:
            points = Point_cloud()

            point_set,center,scale_rate=random_downsample(points,2500)

           
            # if point_set.size()[0] >1000:
            _,pred_choice,points_raw=predictor.get_pred_pointcloud(point_set)

            # pcd_raw = o3d.geometry.PointCloud()
            # pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(points_raw[:, :3]))
            stairs_points = sem_autuomatic.data_align(points_raw, pred_choice)  ###预测结果再整合###
            vis.remove_geometry(stairs_pcd)
            stairs_pcd = sem_autuomatic.viz_pointcloud_o3d(stairs_points)#得到预测后点云
            
            # open3d pcd
            
        


            
            # stairs_pcd.points = o3d.utility.Vector3dVector(stairs_points[:,:3])
           

            # color = [1, 0, 0]  # 红色：[R, G, B]
            # stairs_pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(stairs_points[:,:3]), 1)))


            pred_labels = list(stairs_points[:,3])
           
            max_label = compute_steps_num(pred_labels, 10)
            if max_label!=-1:
                # plane finder
                pcd_plane_fit, plane_params, pcd_stairs_list = plane_finder.get_ransac_pointcloud(stairs_pcd,
                                                                                    list(stairs_points[:, 3]),
                                                                                    display=False)

                pcd_meanpoints_planeparams_points = modelling.get_meanpoints(pcd_plane_fit,pcd_stairs_list,plane_params)
                

                try:
                    idx_list, stairs_params =modelling.modelling_stairs_params(
                        pcd_meanpoints_planeparams_points, center, scale_rate)
                    
                    # print('plane_params',plane_params)
                    normal=plane_params[0][:3]
                    normal1=[]
                    
                    for item in normal:
                        normal1.append(item * -1)
                    
                    
                    # print('normal', normal1)
                    for i in range(min(3, len(idx_list))):
                        idx = idx_list[i]
                        
                        print(f"processing {idx}th step")
                        
                        if not (0 <= idx < 3):
                            print(f"step {idx} is out of range, skip")
                            continue
                        
                        if not pcd_meanpoints_planeparams_points[idx]:
                            print(f"step {idx} is empty, skip")
                            continue
                        
                        pred_step_centrol_point = pcd_meanpoints_planeparams_points[idx][0][:3]
                        print('pred_step_centrol_point',pred_step_centrol_point)
                        
                         # 中心点
                        center_msg = PointStamped()
                        center_msg.header.stamp = rospy.Time.now()
                        center_msg.header.frame_id = 'camera_depth_optical_frame'
                        center_msg.point.x = pred_step_centrol_point[0]
                        center_msg.point.y = pred_step_centrol_point[1]
                        center_msg.point.z = pred_step_centrol_point[2]
                        
                        center_pub.publish(center_msg)
                        

                    # pred_step_centrol_point = pcd_meanpoints_planeparams_points[idx_list[0]][0][:3]
                    
                    
                    
                    if len(normal)!=0:
                        vis.remove_geometry(Arrow)
                        _,Arrow=test1.get_arrow(pred_step_centrol_point,normal1)
                        vis.add_geometry(Arrow)
                    

                except ValueError:  # FPS not passed
                    # print("ValueError")
                    continue
                
                if len(idx_list) <= 1:  # 这时不适合判断
                    # print("Idx list length less than or equal to 1")
                    continue
                
                pred_step_params = stairs_params
                # print('predited params:',pred_step_params)
                
                
                # 法向量
                # stairs_normal = normal1
                # normal_msg = Vector3Stamped()
                # normal_msg.vector.x = stairs_normal[0]
                # normal_msg.vector.y = stairs_normal[1]
                # normal_msg.vector.z = stairs_normal[2]
                
                # normal_pub.publish(normal_msg)


            update_thread = threading.Thread(target=update_visualization(stairs_pcd))
            update_thread.daemon = True
            vis.poll_events()
            vis.update_renderer()
            update_thread.start()

        else:
            print("not points_data")
            time.sleep(2)
        rate.sleep()
