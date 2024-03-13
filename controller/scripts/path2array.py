#!/usr/bin/env python

import rospy
import tf2_ros
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs

# def trajectory_callback(msg):
#     # 获取base坐标系到路径消息中坐标系的转换关系
#     try:
#         transform = tf_buffer.lookup_transform("base", msg.header.frame_id, rospy.Time())
#     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#         rospy.logwarn("Failed to lookup transform: %s", str(e))
#         return

#     # 转换路径消息中的每个姿态到base坐标系下
#     transformed_trajectory = []
#     for pose in msg.poses:
#         transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
#         transformed_trajectory.append(transformed_pose.pose.position.x)
#         transformed_trajectory.append(transformed_pose.pose.position.y)
#         # transformed_trajectory.append(transformed_pose.pose.position.z)

#     # 创建并发布转换后的消息
#     float_array_msg = Float32MultiArray()
#     float_array_msg.data = transformed_trajectory
#     pub.publish(float_array_msg)

def trajectory_callback(msg):
    float_array_msg = Float32MultiArray()
    
    step = 3
    trajectory_data = []
    
    for i in range(15, len(msg.poses), step):
        pose = msg.poses[i]
        trajectory_data.append(pose.pose.position.x)
        trajectory_data.append(pose.pose.position.y)
        
    float_array_msg.data = trajectory_data
    
    pub.publish(float_array_msg)

def trajectory_to_float_array():
    rospy.init_node('trajectory_to_float_array', anonymous=True)

    # 订阅轨迹消息
    rospy.Subscriber("/trajectory", Path, trajectory_callback)

    # 发布 Float32MultiArray 消息
    global pub
    pub = rospy.Publisher("/surf_predict_pub", Float32MultiArray, queue_size=10)

    # 将消息以10Hz的频率发布出去
    # rate = rospy.Rate(10)
    
    # rate.sleep()
    

    # 初始化 tf2
    # global tf_buffer
    # tf_buffer = tf2_ros.Buffer()
    # tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.spin()

if __name__ == '__main__':
    try:
        trajectory_to_float_array()
    except rospy.ROSInterruptException:
        pass
