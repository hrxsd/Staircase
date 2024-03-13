import numpy as np
import open3d as o3d
 
 
def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
 
 
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 
def get_arrow(begin=[0,0,0],vec=[0,0,1]):

    begin = begin
    end = np.add(begin,vec)
    vec_Arr = np.array(end) - np.array(begin)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cone_height=0.025 * 1 ,cone_radius=0.02 * 1,cylinder_height=0.1* 1,cylinder_radius=0.01 * 1,resolution=10)
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()
 
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))
    return mesh_frame, mesh_arrow
 
if __name__ == "__main__":
    mesh_frame, mesh_arrow = get_arrow([1,1,1],[0,0,1])
    o3d.visualization.draw_geometries(
        geometry_list=[mesh_frame,mesh_arrow],
        window_name="after translate", width=800, height=600
    )