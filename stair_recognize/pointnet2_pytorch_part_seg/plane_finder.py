import open3d as o3d
import numpy as np

#分割为N个台阶(先去掉10号，因为10号是非楼梯，所以最大为9号)
#如果是全10点云 不知道情况怎么样
def compute_steps_num(label1,not_step):
    step_idx_list = []
    for i in range(len(label1)):
        if label1[i] != not_step:
            step_idx_list.append(i)
    _set_for_count = set(label1)
    #set_for_count.add(enumrate(tuple(label1)))错误用法
    #print("set_for_count:",set_for_count)
    if not_step in _set_for_count:
        _set_for_count.remove(not_step)
    return int(max(_set_for_count))

#pcd经过采样的  数据的index 早就变了  
def get_group_point_cloud(pcd,labels,class_number):
    point_list_idx = [[] for _ in range(class_number)] #存放不同类点的index
    # pcd_list_idx为二维数组  第一维是分类，第二维是对应点的index
    for idx_point in range(0,len(pcd.points)):
        point_list_idx[int(labels[idx_point])].append(idx_point) # 对应点的label就是__pcd_list中存放的位置的index

    #由point_list_idx导出__pcd_for_ransac
    #pcd_for_ransac有2维，第一维分类，第二维是pcd对象(二维是点个个数，第三维是点的位置)
    pcd_for_ransac = []#组成pcd文件
    for a in range(len(point_list_idx)): #classnumber 既是pcdlistindex的第一维
                                            #在分类维度
        pcd_for_ransac.append(o3d.geometry.PointCloud())
        __point_tmp = []
        for b in range(len(point_list_idx[a])):# 在点云个数维度循环
            __point_tmp.append(pcd.points[point_list_idx[a][b]])

        pcd_for_ransac[a].points = o3d.utility.Vector3dVector(__point_tmp)
    #for i in range(class_number):
        #print("pcd_for_ransac_{}".format(i),len(pcd_for_ransac[i].points))
    return pcd_for_ransac

#功能：点云拟合平面方程
#输入：点云
#输出：__pcd_for_ransac_return:只包含inlier points
#系统超参数：ransac_n：如果一个聚类结果中点的数量小于ransac_n会抛出异常，用以防止该情况。
#           即使点云数量大于该参数，由于拟合会进行降采样（只采用inlier）所以输出的点依然会小于ransace_n
#           这会导致sampling_fps_points跳过。

def get_ransac_pointcloud(pcd,labels,display=False): #class_number 是聚类结果数量
    
    class_number = compute_steps_num(labels,10)+1
    #pcd_for_ransac有2维，第一维分类，第二维是pcd对象(也可以说二维是点个个数，第三维是点的位置)
    pcd_for_ransac = get_group_point_cloud(pcd,labels,12)
    pcd_for_ransac_return = [[] for _ in range(12)] #用来装o3d.geometry.PointCloud()
    ransac_n = 20
    plane_params = [[] for _ in range(12)]
    for idx_class in range(0,class_number-1):
        if len(pcd_for_ransac[idx_class].points) < ransac_n: #如果一个聚类结果中点的数量小于ransac_n会抛出异常，用以防止该情况
            # print("the points in class:%d r below than ransan_c's request"%(idx_class))
            continue
        plane_model, inliers = pcd_for_ransac[idx_class].segment_plane(distance_threshold=0.002,# 内点到平面模型的最大距离
                                        ransac_n = ransac_n,# 用于拟合平面的采样点数
                                        num_iterations=100) # 最大迭代次数
        [a, b, c, d] = plane_model
        plane_params[idx_class] = [-a, -b, -c, -d]
        #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        # 平面内点点云
        inlier_cloud = pcd_for_ransac[idx_class].select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        # 平面外点点云
        # outlier_cloud = pcd_for_ransac[idx_class].select_by_index(inliers, invert=True)
        # outlier_cloud.paint_uniform_color([0, 0, 1.0])
        # 可视化平面分割结果
        if display==True:
            coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
            o3d.visualization.draw_geometries([inlier_cloud]+[coord],
                                                window_name="拟合平面{}".format(idx_class),
                                                width=800,height=600,left =1500,top=500)
        pcd_for_ransac_return[idx_class] = inlier_cloud
    try:
        riser_pcd = pcd_for_ransac[11]
    except IndexError:
        plane_params[11] = [0,0,0,0]
        return pcd_for_ransac_return,plane_params,pcd_for_ransac
    
    if len(riser_pcd.points) != 0:
        riser_array = np.array(riser_pcd.points)
        labels = np.array(riser_pcd.cluster_dbscan(eps=0.07, min_points=10, print_progress=False)) + 1 # for bitcount operation
        max_count= np.argmax(np.bincount(labels + 1))-1
        #labels_count_max_index_in_labels_count = np.argmax(labels_count)
        #labels_count_max_index_in_labels_count -= 1 # invert operation of the previous add
        labels_count_max_index_list_in_pcd = np.where(labels == max_count)
        riser_array = riser_array[labels_count_max_index_list_in_pcd][:]
        if(len(riser_array)>50):
            pcd_riser = o3d.geometry.PointCloud()
            pcd_riser.points = o3d.utility.Vector3dVector(riser_array)
            plane_model_riser, inliers_riser = pcd_riser.segment_plane(distance_threshold=0.002,ransac_n = 50,num_iterations=100) 
            [a, b, c, d] = plane_model_riser
            plane_params[11] = [-a, -b, -c, -d]
        else:
            plane_params[11] = [0,0,0,0]
        # o3d.visualization.draw_geometries([riser_pcd],
        #                                             window_name="拟合平面{}".format(idx_class),
        #                                             width=800,height=600,left =1500,top=500)
    else:
        plane_params[11] = [0,0,0,0]
    return pcd_for_ransac_return,plane_params,pcd_for_ransac
