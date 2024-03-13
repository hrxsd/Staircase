import numpy as np
import os.path

#用来判断nan用到math
import math
import sys
#
sys.path.append("~/pointnet2_pytorch_part_seg")
import shutil
import random
#from Lib.visualizelib import *

######################常用函数#############################
#分割为N个台阶(先去掉10号，因为10号是非楼梯，所以最大为9号)
#如果是全10点云 不知道情况怎么样
#
###########################################################
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


#####################标注法向量####################
#功能：给CC算全部法向量的点云文件中计算出平均的法向量保存
#输入：点云文件
#输出：原点云文件，法向量文件
#注意：我这里还忘了法向量的标号，比如该法向量是属于1还是2楼梯,或许set默认排好了，对的已经排好序了
##################################################
def add_normals(dictOrder,fileOrder):
    
    #内参
    not_step = 10

    #点云预处理
    rootpath = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled\{}'
    read_fileroot = '{stair}_{order}_withnorms.txt'
    read_datapath = os.path.join(rootpath.format(dictOrder),read_fileroot.format(stair = dictOrder,order = fileOrder))
    #points   = np.loadtxt(str(datapath),usecols=(0,1,2)).astype(np.float32)
    #rgb      = np.loadtxt(str(datapath),usecols=(3,4,5))
    label   = np.loadtxt(str(read_datapath),usecols=(6)).astype(np.float32) #float 才能把-2147483648 识别成Nan
    normals  = np.loadtxt(str(read_datapath),usecols=(7,8,9)).astype(np.float32)
    
    max_step_label = compute_steps_num(label,not_step)
    #格式为 max_step_label*相应点的长度的idx列表
    step_idx_list = [[] for _ in range(max_step_label+1)]
    #print(step_idx_list)
    for i in range(len(label)):
        if label[i] != not_step:#跳过防止outofindex
            step_idx_list[int(label[i])].append(i) #这里隐含着排序信息。用label的值变成了列表的idx
        else:
            continue
    #print("step_idx_list:",step_idx_list)

    ################每一个台阶算法向量###################
    
    #保存平均向量用
    average_normals_list = [[0,0,0] for _ in range(len(step_idx_list))]
    #print("average_normals_list:",average_normals_list)
    #normals 等同于t_normals
    #预处理法向量列表，使其能够运算,这里是所有的normals都在，即使是notstep
    t_normals = [[[],[],[]] for _ in range(len(normals))]
    for j in range(len(normals)):
        t = list(normals[j]) #不换成list不然没法识别numpyndarray之间的空格
        t_normals[j][0] = t[0]
        t_normals[j][1] = t[1]
        t_normals[j][2] = t[2]
    #print("t_norms:",t_normals)

    for step_order in range(len(step_idx_list)):
        for point_order in step_idx_list[step_order]:
            average_normals_list[step_order][0] += t_normals[point_order][0]/len(step_idx_list[step_order])
            average_normals_list[step_order][1] += t_normals[point_order][1]/len(step_idx_list[step_order])
            average_normals_list[step_order][2] += t_normals[point_order][2]/len(step_idx_list[step_order])
        #average_normals_list[step_order] = average_normals_list[step_order]/len(step_idx_list[step_order])
    #print("average_normals_list:",average_normals_list)
        #average_normals_list[step_order] = average_normals_list[step_order]/len(step_idx_list[step_order])
    
    #标注
    savedata = np.asarray(average_normals_list)
    write_fileroot = '{stair}_{order}_norm.txt'
    write_datapath = os.path.join(rootpath.format(dictOrder),write_fileroot.format(stair = dictOrder,order = fileOrder))
    #print(savedata)
    np.savetxt(write_datapath,savedata,fmt="%.8f %.8f %.8f") #float位数太多    


#####################标注中心点####################
#功能：从points中计算出点云的中心点并保存在文件中
#输入：点云文件
#输出：点云中心点文件
#注意：比如中心点是属于1还是2楼梯,或许set默认排好了，对的已经排好序了
##################################################
def add_centrol_points(rootpath,dictOrder,fileOrder):
    
    # 内参
    not_step = 10

    # 点云读取保存路径
    read_fileroot = '{}\{}_{}.txt'
    read_datapath = os.path.join(rootpath,read_fileroot.format(dictOrder,dictOrder,fileOrder))
    write_fileroot = '{}\{}_{}_centrol.txt'
    write_datapath = os.path.join(rootpath,write_fileroot.format(dictOrder,dictOrder,fileOrder))

    # 点云数据读取
    points   = np.loadtxt(str(read_datapath),usecols=(0,1,2)).astype(np.float32)
    #rgb      = np.loadtxt(str(read_datapath),usecols=(3,4,5))
    label   = np.loadtxt(str(read_datapath),usecols=(6)).astype(np.float32) #float 才能把-2147483648 识别成Nan
    #normals  = np.loadtxt(str(read_datapath),usecols=(7,8,9)).astype(np.float32)
    
    # 计算楼梯数量
    max_step_label = compute_steps_num(label,not_step)
    #格式为 max_step_label*相应点的长度的idx列表
    step_idx_list = [[] for _ in range(max_step_label+1)]
    # 初始化楼梯分类索引列表
    for i in range(len(label)):
        if label[i] != not_step:#跳过防止outofindex
            step_idx_list[int(label[i])].append(i) #这里隐含着排序信息。用label的值变成了列表的idx
        else:
            continue
    #print("step_idx_list:",step_idx_list)

    ################每一个台阶算中心点###################
    
    # 保存中心点用
    average_center_list = [[0,0,0] for _ in range(len(step_idx_list))]
    # 暂存点云位置
    t_points = [[[],[],[]] for _ in range(len(points))]
    for j in range(len(points)):
        t = list(points[j]) #不换成list不然没法识别numpyndarray之间的空格
        t_points[j][0] = t[0]
        t_points[j][1] = t[1]
        t_points[j][2] = t[2]

    for step_order in range(len(step_idx_list)):
        for point_order in step_idx_list[step_order]:
            average_center_list[step_order][0] += t_points[point_order][0]/len(step_idx_list[step_order])
            average_center_list[step_order][1] += t_points[point_order][1]/len(step_idx_list[step_order])
            average_center_list[step_order][2] += t_points[point_order][2]/len(step_idx_list[step_order])

    #标注
    savedata = np.asarray(average_center_list)
    np.savetxt(write_datapath,savedata,fmt="%.8f %.8f %.8f") #float位数太多    


#####################生成数据的方法#########################
#功能：从台阶点云中随机旋转再用ransac算出法向量再标记
#输入：台阶碎片
#输出：法向量标记文件
###########################################################
#TODO

##########生成测试ThomasWestFechtal数据的方法################
#功能：把点云数据旋转
#输入：评估的点云数据（不必用训练集）
#输出：ThomasWestfechtal能用的点云数据
###########################################################
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

def thomas_pointcloud_generate(rootpath,dictOrder,fileOrder):

    # 内参
    not_step = 10
    # 定义旋转参数
    rotation_angle =  -(np.pi / 2)  # 旋转角度（以弧度为单位）
    rotation_axis = np.array([1, 0, 0])  # 旋转轴的XYZ分量

    # 点云读取保存路径
    read_fileroot = '{}\{}_{}.txt'
    read_datapath = os.path.join(rootpath,read_fileroot.format(dictOrder,dictOrder,fileOrder))
    write_fileroot = '{}\{}_{}_thomas.txt'
    write_datapath = os.path.join(rootpath,write_fileroot.format(dictOrder,dictOrder,fileOrder))

    # 点云数据读取
    points   = np.loadtxt(str(read_datapath),usecols=(0,1,2)).astype(np.float32)
    rgb      = np.loadtxt(str(read_datapath),usecols=(3,4,5))
    label   = np.loadtxt(str(read_datapath),usecols=(6)).astype(np.float32) #float 才能把-2147483648 识别成Nan
    #normals  = np.loadtxt(str(read_datapath),usecols=(7,8,9)).astype(np.float32)

    # 进行旋转变换
    rotated_points = rotate_point_cloud(points, rotation_angle, rotation_axis)

    # 输出旋转后的点云坐标
    # print(rotated_points)
    
    # 旋转后的文件保存
    savedata = np.concatenate((rotated_points,rgb,label.reshape(-1,1)),axis = 1)
    np.savetxt(write_datapath,savedata,fmt="%.8f %.8f %.8f %d %d %d %d") #float位数太多    


#####################生成数据的方法#########################
#功能：数据集每种类别中随机挑选6个生成验证集
#输入：标注好的文件
#输出：随机挑选的6个文件
###########################################################

def random_select(num):
    
    # 内建函数
    def append_text_file(file_name, text):
        with open(file_name, 'a') as file:
            file.write(text + '\n')

    #内参

    #点云存储路径 
    rootpath = r'D:\Project_on_going\StairsSet\SZTUstairs\dataset_withlabell'

    #数子文件夹中有多少个文件
    def count_files_in_subfolders(folder_path):
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        subfolders.sort(key=lambda x: int(os.path.basename(x)))
        result = {}
        for subfolder in subfolders:
            file_count = len(os.listdir(subfolder))
            result[os.path.basename(subfolder)] = file_count
        return result

    counted_files = count_files_in_subfolders(rootpath)
    
    
    #每个文件夹中文件数量生成num个待选文件列表
    for i in range(1,11):
        random_number_list = random.sample(range(1,counted_files["{}".format(i)]+1), num)#随机数产生 在每个文件夹种的文件多少为边界
        for j in random_number_list:
            shutil.move(rootpath+'\{}\{}_{}.txt'.format(i,i,j),'D:\Project_on_going\StairsSet\SZTUstairs\eval_dataset')
            append_text_file('D:\Project_on_going\StairsSet\SZTUstairs\eval_dataset\\relationship_train_eval.txt',"{}_{}".format(i,j))


####################点云数据整理############################
#功能：预处理，去除点云行间乱码
#输入：预处理点云txt的绝对路径
#输出：某点云中的错误位置的行号，以异常报出
##########################################################
def count_files_in_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort(key=lambda x: int(os.path.basename(x)))

    result = {}
    for subfolder in subfolders:
        file_count = len(os.listdir(subfolder))
        result[os.path.basename(subfolder)] = file_count

    return result

def get_error_pointCloud_txt(root):
           
    #完整路径由 root + dictOrder + stairs(same as dicrOrder) + filesOrder 组成
    datapath = {} #文件序数和文件完整路径对应字典
    dictNumber = 10
        
    fileslist = [] 
    file_counts = count_files_in_subfolders(root)
    for k,v in file_counts.items():
        fileslist.append(v)
    # fileslist = [22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
    t = 0
    tmp = []
    tmp.append(0)
    for k,v in file_counts.items():
        t += v
        tmp.append(t)
    # tmp = [0, 22, 92, 40, 56, 71, 64, 65, 60, 67, 74]
    #字典赋值
    #法向量标注文件查找字典的赋值
    for dictOrder in range(dictNumber): #dictOrder 0,1,2,3,4,5,6,7,8,9
        for filesOrder in range(0,fileslist[dictOrder]):#filesOrder 0到当前dictorder的长度
            datapath[filesOrder+tmp[dictOrder]] = os.path.join(root,str(dictOrder+1),'{}_{}.txt'.format(dictOrder+1,filesOrder+1))
    for i in range(len(datapath)):
        print("files:{}".format(datapath[i]))
        pointSet     = np.loadtxt(str(datapath[i]),usecols=(0,1,2)).astype(np.float32)     # 读取点云
        pointRGB     = np.loadtxt(str(datapath[i]),usecols=(3,4,5)).astype(np.int64)       # 读取RGB
        pointLabel   = np.loadtxt(str(datapath[i]),usecols=(6)).astype(np.int64)           # 读取标签1


###################点云数据整理############################
#功能:预处理，删除点云第几列
#输入:预处理点云txt的绝对路径
#输出:txt点云
###########################################################

def remove_coloum(dictOrder,fileOrder):

    rootpath = 'D:\Project_on_going\StairsSet\SZTUstairs\dataset_withlabell_extract_eval_downsampled_normal\{}'
    fileroot = '{stair}_{order}.txt'
    datapath = os.path.join(rootpath.format(dictOrder),fileroot.format(stair = dictOrder,order = fileOrder))
    
    points = np.loadtxt(str(datapath),usecols=(0,1,2)).astype(np.float32)
    rgb    = np.loadtxt(str(datapath),usecols=(3,4,5))
    label1 = np.loadtxt(str(datapath),usecols=(6)).astype(np.float32) #float 才能把-2147483648 识别成Nan
    remove_coloum = np.loadtxt(str(datapath),usecols=(7)).astype(np.float32) 

    savedata = np.concatenate((points,rgb,label1.reshape(-1,1)),axis = 1)
    np.savetxt(datapath,savedata,fmt="%.8f %.8f %.8f %d %d %d %d") #float位数太多  


###########################################################
#注意:事实证明这个方法不可靠，open3d pcd文件不能添加自定义属性
#功能：想要对于点云进行空间上的降采样 但是事实证明不可行
#
###########################################################

def downsample_point_cloud(dictOrder,fileOrder,target_points):
    
    # 路径
    rootpath = 'D:\Project_on_going\StairsSet\SZTUstairs\dataset_withlabell_eval_downsampled\{}'
    read_file_path = '{stair}_{order}.txt'
    write_file_path = '{stair}_{order}_downsampled.txt'
    read_datapath = os.path.join(rootpath.format(dictOrder),read_file_path.format(stair = dictOrder,order = fileOrder))
    write_datapath = os.path.join(rootpath.format(dictOrder),write_file_path.format(stair = dictOrder,order = fileOrder))
    # 读取
    points = np.loadtxt(str(read_datapath),usecols=(0,1,2)).astype(np.float32)
    rgb    = np.loadtxt(str(read_datapath),usecols=(3,4,5)).astype(np.int32)
    label1 = np.loadtxt(str(read_datapath),usecols=(6)).astype(np.int32) #float 才能把-2147483648 识别成Nan
    
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb) 

    # 添加标签信息作为点云属性
    label_property = o3d.utility.IntVector(label1)
    pcd.points["label"] = label_property

    # 进行空间均匀降采样
    downsampled_cloud = pcd.voxel_down_sample(voxel_size=target_points/len(pcd.points))

    # 提取降采样后的点云数据
    downsampled_positions = np.asarray(downsampled_cloud.points)
    downsampled_colors = np.asarray(downsampled_cloud.colors)
    downsampled_labels = np.asarray(downsampled_cloud.label)

    # 保存降采样后的点云数据
    data = np.concatenate((downsampled_positions, downsampled_colors, downsampled_labels), axis=1)
    np.savetxt(write_datapath, data, fmt='%f %f %f %d %d %d %d')


##########################数据预处理#################################
#功能:墙壁和竖直面未做标注，这里可以添加任意label 给他
#
#
###########################################################
def Add_labels(dictOrder,fileOrder):

    rootpath = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled\{}'
    fileroot = '{stair}_{order}.txt'

    datapath = os.path.join(rootpath.format(dictOrder),fileroot.format(stair = dictOrder,order = fileOrder))
    points = np.loadtxt(str(datapath),usecols=(0,1,2)).astype(np.float32)
    rgb    = np.loadtxt(str(datapath),usecols=(3,4,5))
    label1 = np.loadtxt(str(datapath),usecols=(6)).astype(np.float32) #float 才能把-2147483648 识别成Nan
    #print("label:",label1)
    
    tmplist = [] #用以保存预更换标签列的第几行（也就是行序数）
    for i in range(len(label1)):
        # if label1[i] == -1:
        if math.isnan(label1[i]):
            tmplist.append(i)
    print(tmplist)
    label1[tmplist] = 10

    savedata = np.concatenate((points,rgb,label1.reshape(-1,1)),axis = 1)
    np.savetxt(datapath,savedata,fmt="%.8f %.8f %.8f %d %d %d %d") #float位数太多


if __name__ == '__main__':
    
    rootpath = 'D:\Project_on_going\pointnet2_pytorch_part_seg\dataset\dataset_withlabell_eval_thomas'
    dictNum = 10
    filesNum = 6
    for i in range(1,dictNum+1):
        for j in range(1,filesNum+1):
            print(i,' ',j)
            thomas_pointcloud_generate(rootpath=rootpath,dictOrder=i,fileOrder=j)
