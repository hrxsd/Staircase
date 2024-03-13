import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import random
import sys
import os

sys.path.append("D:\Project_on_going\pointnet2_pytorch_part_seg")


def high_level_pcd(pcd,label):
    
    app = gui.Application.instance
    app.initialize()
    
    #可视化 O3dvisulizer 只能更新tpointcloud的显示 
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("PointsCLoud", pcd)


    for idx_class in range(0,len(pcd.points)):
        vis.add_3d_label(pcd.points[idx_class], "{}".format(label[idx_class]))

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
    
   
#输入不是PCD点云 
def high_level(points,label):#point是list，label是tensor
    
    app = gui.Application.instance
    app.initialize()
    
    #测试输入
    #points = make_point_cloud(100, (0, 0, 0), 1.0)

    #输入不是pcd点云 所以先变成pcd点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.random.uniform(0.0, 1.0, size=[len(label), 3])
    pcd.colors = o3d.utility.Vector3dVector(colors)  #输入点云类别*3的矩阵
    
    #可视化
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("PointsCLoud", pcd)

    #for idx_class in range(0,np.size(label,0)): #多少类别 {[1,2,3],[546,564],[7]...}这种
    #    #print(idx_class)
    #    for idx_point in range(0,np.size(label[idx_class][:])): # 一个类有多少个点
    #        vis.add_3d_label(pcd.points[label[idx_class][idx_point]], "{}".format(idx_class))

    for idx in range(0,label.shape[0]):
        vis.add_3d_label(pcd.points[idx][:], "{}".format(label[idx][0]))
        
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

def viz_pointcloud_normals_o3d(points,normals):
    
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    coord= o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.5, 0.5, 0])
    o3d.visualization.draw_geometries([pcd]+[coord], window_name="法向量显示结果输出",point_show_normal=True,width=800,  height=600)  



""" def low_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.Material()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, len(points.points)):
        widget3d.add_3d_label(points.points[idx], "{}".format(idx))
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()
 """
if __name__ == "__main__":
    root = 'D:\Project_on_going\StairsSet\SZTUstairs\\near_labelled'
    dictOrder = 10
    filesOrder = 5
    datapath = os.path.join(root,str({}).format(dictOrder),'{}_{}_withnorms.txt'.format(dictOrder,filesOrder))
    pointSet     = np.loadtxt(str(datapath),usecols=(0,1,2)).astype(np.float32)     # 读取点云
    pointNormals = np.loadtxt(str(datapath),usecols=(7,8,9)).astype(np.float32)         # 读取法向量
    viz_pointcloud_normals_o3d(points=pointSet,normals=pointNormals)
    


# visualizer 不能显示标签文字
# O3dVisualizer 只能更新tpointcloud的显示  update geometry 那个
# gui显示的话 visualizer createwindow不生成guiapp中用的窗口
