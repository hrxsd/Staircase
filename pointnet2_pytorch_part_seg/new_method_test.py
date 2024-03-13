import numpy as np
import open3d as o3d

##################################################################################################################
# #实验记录用数据结构
# experiment_data = {
#     '日期':[],
#     '版本':[],
#     '数据集':[],
#     'index':[],
#     '网络预测时间':[],
#     '平面计算时间':[],
#     '中心点计算时间':[],
#     '参数计算时间':[]
#     }
# def record(key,value):
#     experiment_data[key].append(value)
 
# if __name__ == '__main__':
#     for i in range(5):
#         record('日期',12)
#         print(experiment_data)

##################################################################################################################

# import numpy as np

# def rotate_point_cloud(points, angle, axis):
#     """
#     旋转点云
    
#     参数：
#     points: numpy数组，形状为(N, 3)，表示点云中的N个点的XYZ坐标
#     angle: float，旋转角度（以弧度为单位）
#     axis: numpy数组，形状为(3,)，表示旋转轴的XYZ分量
    
#     返回值：
#     numpy数组，形状为(N, 3)，表示旋转后的点云坐标
#     """
#     cos_theta = np.cos(angle)
#     sin_theta = np.sin(angle)
#     rotation_matrix = np.array([[cos_theta + axis[0] ** 2 * (1 - cos_theta), 
#                                  axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta, 
#                                  axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
#                                 [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta, 
#                                  cos_theta + axis[1] ** 2 * (1 - cos_theta), 
#                                  axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
#                                 [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta, 
#                                  axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta, 
#                                  cos_theta + axis[2] ** 2 * (1 - cos_theta)]])
#     rotated_points = np.dot(points, rotation_matrix.T)
#     return rotated_points

# # 读取点云数据
# file_path = 'point_cloud.txt'  # 替换为你的文件路径
# points = np.loadtxt(file_path)

# # 定义旋转参数
# rotation_angle = np.pi / 4  # 旋转角度（以弧度为单位）
# rotation_axis = np.array([0, 1, 0])  # 旋转轴的XYZ分量

# # 进行旋转变换
# rotated_points = rotate_point_cloud(points, rotation_angle, rotation_axis)

# # 输出旋转后的点云坐标
# print(rotated_points)

##################################################################################################################

