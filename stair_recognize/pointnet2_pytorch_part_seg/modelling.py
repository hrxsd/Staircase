import numpy as np
import open3d as o3d
from robot_math import *

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):

    average_data = np.mean(data,axis=0)       #求 NX3 向量的均值
    decentration_matrix = data - average_data   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)  #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量
    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


#功能：从点云中三维重建出拟合的平面并且返回平面中心点
#输入：pcdlist pcdraw
#输出：一个列表：meanpoints:[pcd_mean_point,pcd_list[pcd_idx],mesh]
def get_meanpoints(pcd_list,pcd_stairs_list,plane_params,sampling_fps_point = 20):
        meanpoints = [[] for _ in range(len(pcd_list))] #返回存放全部的中心点和楼梯碎片用以碎片补偿

        # the last one element is the riser. Need a process that is not like this one.
        for idx in range(len(pcd_list)-1): 
            try:
                if len(pcd_list[idx].points) < sampling_fps_point:
                    # print("the points in this plane cannot reached the least points requset of FPS")
                    continue
            except AttributeError:
                continue
            #中心点         
            mean_ndarray = np.mean(np.asarray(pcd_list[idx].points),axis=0) #返回1*3
            #存储单个中心点用
            # pcd_mean_point = o3d.geometry.PointCloud()
            # pcd_mean_point.points = o3d.utility.Vector3dVector(mean_ndarray.reshape(1,3))
            # pcd_mean_point.paint_uniform_color([0, 0, 1])
            #重建
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_list[pcd_idx], alpha=2)
            #可视化
            # coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
            # o3d.visualization.draw_geometries([pcd_mean_point,_pcd_raw,_mesh]+[coord],
            #                                     window_name="三维重建和中心点{}".format(_pcd_idx),
            #                                     point_show_normal=False,
            #                                     width=800,height=600,left =1500,top=500,
            #                                     mesh_show_wireframe=False,mesh_show_back_face=True)
            # True的话函数会构建返回值
            # pcd_meanpoints_fragments[pcd_idx] = [mean_ndarray,pcd_list[pcd_idx],mesh,plane_params[pcd_idx]]
            meanpoints[idx] = [mean_ndarray,plane_params[idx],pcd_list[idx].points,pcd_stairs_list[idx].points]       
        #print("pcd_meanpoints_fragments",len(_pcd_meanpoints_fragments))
        meanpoints[11] = [[],plane_params[11],[],[]]
        return meanpoints  



#功能：从中心点求解出楼梯高度和宽度
#输入：重建之后得出的中心点
#输出：实际的距离（以米为单位）
def modelling_stairs_params(pcd_meanpoints_planeparams_points,center,scale_rate,depthscale = 2500):

    # inner func
    def numpy_flat(a):
        return list(np.array(a).flat)
    stairs_params = [[],[]]#height，depth
   
    # find not empty idx
    tmp_list = []
    for i in range(len(pcd_meanpoints_planeparams_points)):
        if len(pcd_meanpoints_planeparams_points[i]) != 0:
            tmp_list.append(i)
    
    # plane with one element will cause a Nonetype exception. 
    # return it for using to step over this evaluated file
    if(len(tmp_list)) <= 2:
        return tmp_list,stairs_params
    
    # distance between adjancy steps
    tmp_list_for_idx_distance = []
    for j in range(len(tmp_list)-1):
        tmp_list_for_idx_distance.append(tmp_list[j+1] - tmp_list[j])
    
    # data extract
    first_step_centrol = pcd_meanpoints_planeparams_points[tmp_list[0]][0]
    first_step_normal = pcd_meanpoints_planeparams_points[tmp_list[0]][1][:3]
    second_step_centrol = pcd_meanpoints_planeparams_points[tmp_list[1]][0]
    prev = first_step_centrol
    next = second_step_centrol
    
    pre_x = numpy_flat(np.asarray(prev))[0]
    pre_y  = numpy_flat(np.asarray(prev))[1]
    pre_z  = numpy_flat(np.asarray(prev))[2]
    next_x = numpy_flat(np.asarray(next))[0]
    next_y = numpy_flat(np.asarray(next))[1]
    next_z = numpy_flat(np.asarray(next))[2]
    
    pre = np.empty([1,3], dtype = float)
    next = np.empty([1,3], dtype = float)
    pre[0][0] = pre_x
    pre[0][1] = pre_y
    pre[0][2] = pre_z
    next[0][0] = next_x
    next[0][1] = next_y
    next[0][2] = next_z

    # correct process
    # get a rotation matrix of height and depth, height is the first to get an answer, making sure of that.
    rotate_matrix_h = rotmat_between_vectors(first_step_normal,[0,-1,0])
    
    if pcd_meanpoints_planeparams_points[11][1][:3] == [0,0,0]:
        rotated_points = (rotate_matrix_h @ np.array(pcd_meanpoints_planeparams_points[tmp_list[1]][3]).transpose(1,0)).transpose(1,0)
        # use PCA to get target vector
        _,eigen_vector = PCA(rotated_points)
        width_ventor = eigen_vector[0]
        target_depth_vector = np.cross(first_step_normal,width_ventor)
        target_depth_vector = np.array([0,0,target_depth_vector[2]])
    else:
        # another way to get target vector
        target_depth_vector = pcd_meanpoints_planeparams_points[11][1][:3]/np.linalg.norm(pcd_meanpoints_planeparams_points[11][1][:3])
        # 旋转后的Z轴再转向PCA最大与first_step_normal 之间的叉乘方向
        # rotate_matrix_d = rotmat_between_vectors(target_depth_vector-np.array(rotated_central_point),[0,0,-1])
    rotate_matrix_d = rotmat_between_vectors(target_depth_vector,[0,0,-1])
    
    pre_correct =  (rotate_matrix_h @ pre.transpose(1,0))*scale_rate #+ center
    next_correct = (rotate_matrix_h @ next.transpose(1,0))*scale_rate #+ center

    # height
    # stairs_params[0] = (abs(pre_y - next_y)*depthscale/1000)/tmp_list_for_idx_distance[0]
    stairs_params[0] = (abs(pre_correct[1][0] - next_correct[1][0])*depthscale/1000)/tmp_list_for_idx_distance[0]
    
    # pre_correct =  (rotate_matrix_d @ pre.transpose(1,0))*scale_rate #+ center
    # next_correct = (rotate_matrix_d @ next.transpose(1,0))*scale_rate #+ center
    pre_correct =  rotate_matrix_d @ pre_correct
    next_correct = rotate_matrix_d @ next_correct
     
    # depth
    # stairs_params[1] = (abs(pre_z - next_z)*depthscale/1000)/tmp_list_for_idx_distance[0]
    stairs_params[1] = (abs(pre_correct[2][0] - next_correct[2][0])*depthscale/1000)/tmp_list_for_idx_distance[0]

    # usually do not choose the first step, it's shape are not convinent for find a pca solution. idx2 pass through the filter of RANSAC, 
    # idx 3 is not.
    # rotated_points = (rotate_matrix_d @ rotate_matrix_h @ np.array(pcd_meanpoints_planeparams_points[tmp_list[1]][3]).transpose(1,0)).transpose(1,0)
    # rotated_central_point = rotate_matrix_d @ rotate_matrix_h @ second_step_centrol
    
    return tmp_list,stairs_params

