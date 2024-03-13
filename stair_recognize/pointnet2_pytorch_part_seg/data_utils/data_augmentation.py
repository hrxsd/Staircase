import numpy as np
import random 
import os
import open3d as o3d

def random_remove_steps(pointLabelClass, steps_remove):
    population = set()
    range_end = pointLabelClass
    mu = int(pointLabelClass/2)
    sigma = int(pointLabelClass/2)
    while len(population) < steps_remove:
        num = int(random.gauss(mu, sigma))
        if 0 <= num <= range_end:
            population.add(num)
    return list(population)

def random_scramble_point_order(points,labels):

    # Step 1: 创建索引映射
    num_points = len(points)
    index_mapping = np.arange(num_points)

    # Step 2: 打乱索引映射
    np.random.shuffle(index_mapping)

    # Step 3: 重新排列点云和标签
    shuffled_points = points[index_mapping]
    shuffled_labels = labels[index_mapping]

    return shuffled_points,shuffled_labels

def random_gaussian_remove_point(point,label):
    
    label_set = set(label)
    #这个前提是排序好
    #TODO 一旦排序错误引发问题
    label_list = sorted(list(label_set))
    #print(label_list)

    #按label分组点云的index
    point_grouped = [[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(point)):
        point_grouped[label[i]].append(i)

    #确定保留的元素个数
    remain_list =  []
    for i in range(len(point_grouped)):
        point_grouped_len = len(point_grouped[i])
        if(point_grouped_len) == 0:
            continue
        point_grouped_len *= 0.8
        remain_num = abs(int(np.random.normal(point_grouped_len,10)))
        if remain_num > len(point_grouped[i]):
            remain_num = len(point_grouped[i])
            remain_list.append(remain_num)
        else:
            remain_list.append(remain_num)

    #确定保留的元素
    #这里需要些加入超过原有数量怎么办的代码
    remain_point_index_list = []
    for i in range(len(remain_list)):
        remain_point_index_list.append(random.sample(point_grouped[label_list[i]],remain_list[i]))
    
    

    def flatten(li):
        return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])
    
    remain_point_index_list = flatten(remain_point_index_list)
    
    label = label[remain_point_index_list]
    point = point[remain_point_index_list,:3]
    return point,label

# def random_remove_not_step_points(k):
#     random_num = np.random.normal(0,1)
#     if random_num > k:
#         return 1
#     else:
#         return 0 

def random_remove_not_step_points(k):
    random_num = random.uniform(0,1)
    if random_num > k:
        return 1
    else:
        return 0 

# def random_remove_ball_points(pointCloud,pointlabel,rrb_radius_distribution = 0.1):
#     #random_radius 得考虑到台阶的最大范围，后期进行调参
#     random_radius = abs(np.random.normal(0,rrb_radius_distribution))
#     #random_centrol 得考虑到点云的最大范围。后期进行自动化测试
#     random_centrol = np.random.normal(0,1,size=3)
#     new_point_cloud_idx = []
#     for idx in range(len(pointlabel)):
#         # 计算点与球心的距离
#         distance = np.linalg.norm(pointCloud[idx] - random_centrol)
#         # 检查点是否在球的外部
#         if distance > random_radius:
#             new_point_cloud_idx.append(idx)
#     return new_point_cloud_idx

# 靠中间的点是最重要的
# 所以坐标采用正态分布
def random_remove_ball_points(pointCloud,pointlabel,rrb_radius_max = 0.1):
    #random_radius 得考虑到台阶的最大范围，后期进行调参
    random_radius = random.uniform(0,rrb_radius_max)
    #random_centrol 得考虑到点云的最大范围。后期进行自动化测试
    random_centrol = np.random.normal(0,1,size=3)
    new_point_cloud_idx = []
    for idx in range(len(pointlabel)):
        distance = np.linalg.norm(pointCloud[idx] - random_centrol)
        if distance > random_radius:
            new_point_cloud_idx.append(idx)
    return new_point_cloud_idx

def random_guassian_move(pointCloud,rgm_step_divetion):
    #divetion = 0.002
    for idx in range(len(pointCloud)):
        random_shift_x = np.random.normal(0,rgm_step_divetion)
        random_shift_y = np.random.normal(0,rgm_step_divetion)
        random_shift_z = np.random.normal(0,rgm_step_divetion)
        pointCloud[idx][0] += random_shift_x
        pointCloud[idx][1] += random_shift_y
        pointCloud[idx][2] += random_shift_z
    
    return pointCloud

def test_function(dictOrder,fileOrder):

    rootpath = 'D:\Project_on_going\pointnet2_pytorch_part_seg\dataset\dataset_withlabell_eval\{}'
    fileroot = '{stair}_{order}.txt'
    datapath = os.path.join(rootpath.format(dictOrder),fileroot.format(stair = dictOrder,order = fileOrder))
    
    points = np.loadtxt(str(datapath),usecols=(0,1,2)).astype(np.float32)
    points = random_guassian_move(points,0.002)

    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(points[:,:3]))
    o3d.visualization.draw_geometries([pcd_raw], window_name='随机高斯移动测试',
                                        point_show_normal=False,
                                        width=800,height=600,left = 1500,top=500)  



if __name__ == '__main__':

    for i in range(1,11):
        for j in range(1,7):
            test_function(i,j)
