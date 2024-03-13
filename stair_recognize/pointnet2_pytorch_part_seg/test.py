import numpy as np

def extract_point_cloud_by_category(point_cloud, seg_pred, category):
    # 寻找与给定类别匹配的点云索引
    category_indices = np.where(seg_pred != category)[0]
    
    # 提取相应类别的点云
    category_point_cloud = point_cloud[category_indices]
    
    return category_point_cloud,category_indices

# 假设point_cloud是包含所有点云的numpy数组，每个点云有3个坐标值
point_cloud = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]])

# 假设seg_pred是包含seg_pred结果的numpy数组，表示每个点云的类别
seg_pred = np.array([0, 0, 1, 0])

# 提取类别为1的点云
category_1_point_cloud = extract_point_cloud_by_category(point_cloud, seg_pred, 1)
print("Category 1 point cloud:")
print(category_1_point_cloud)
