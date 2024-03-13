#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

ros::Publisher transformed_center_pub;
// ros::Publisher transformed_normal_pub;
tf2_ros::Buffer tfBuffer;
geometry_msgs::PoseArray pose_array;

void centerCallback(const geometry_msgs::PointStamped::ConstPtr& original_center)
{
    try
    {   
        geometry_msgs::TransformStamped transformStamped = tfBuffer.lookupTransform("odom", "front_camera_depth_optical_frame", ros::Time(0));

        geometry_msgs::PointStamped transformed_center;
        // tfBuffer.transform(*original_center, transformed_center, "odom");
        
        tf2::doTransform(*original_center, transformed_center, transformStamped);

        transformed_center.header.frame_id = "odom";

        // 创建一个PoseStamped类型的消息
        geometry_msgs::PoseStamped transformed_center_pose;
        // 将transformed_center的数据赋值给transformed_center_pose
        transformed_center_pose.header = transformed_center.header;
        transformed_center_pose.pose.position.x = transformed_center.point.x;
        transformed_center_pose.pose.position.y = transformed_center.point.y;
        transformed_center_pose.pose.position.z = transformed_center.point.z;

        tf2::Quaternion q;
        q.setRPY(0, -1.57, 0);
        transformed_center_pose.pose.orientation = tf2::toMsg(q);

        // 将transformed_center_pose添加到pose_array中
        pose_array.poses.push_back(transformed_center_pose.pose);

        // 检查是否已经收集到三个台阶的中心点
        if (pose_array.poses.size() == 3)
        {
            pose_array.header.frame_id = "odom";
            pose_array.header.stamp = ros::Time::now();

            // 发布pose_array
            transformed_center_pub.publish(pose_array);

            ROS_INFO("Transformed stair1 center: x=%f, y=%f, z=%f in frame %s", 
                    pose_array.poses[0].position.x, pose_array.poses[0].position.y, pose_array.poses[0].position.z, pose_array.header.frame_id.c_str());
            ROS_INFO("Transformed stair2 center: x=%f, y=%f, z=%f in frame %s", 
                    pose_array.poses[1].position.x, pose_array.poses[1].position.y, pose_array.poses[1].position.z, pose_array.header.frame_id.c_str());
            ROS_INFO("Transformed stair3 center: x=%f, y=%f, z=%f in frame %s", 
                    pose_array.poses[2].position.x, pose_array.poses[2].position.y, pose_array.poses[2].position.z, pose_array.header.frame_id.c_str());
            ROS_INFO("============================================================");

            // 清空pose_array
            pose_array.poses.clear();
        }
    }
    catch(tf2::TransformException& ex)
    {
        ROS_WARN("Failed to transform center: %s", ex.what());
    }
    
}

// void normalCallback(const geometry_msgs::Vector3Stamped::ConstPtr& original_normal)
// {
//     try
//     {
//         geometry_msgs::TransformStamped transformStamped = tfBuffer.lookupTransform("odom", "camera_depth_optical_frame", ros::Time(0));

//         geometry_msgs::Vector3Stamped transformed_normal;
//         // tfBuffer.transform(*original_normal, transformed_normal, "odom");

//         tf2::doTransform(*original_normal, transformed_normal, transformStamped);
//         transformed_normal.header.frame_id = "odom";

//         transformed_normal_pub.publish(transformed_normal);

//         ROS_INFO("Transformed Normal: (x=%f, y=%f, z=%f) in frame %s", 
//             transformed_normal.vector.x, transformed_normal.vector.y, transformed_normal.vector.z, transformed_normal.header.frame_id.c_str());
//         ROS_INFO("Original Normal: (x=%f, y=%f, z=%f) in frame camera_depth_optical_frame", 
//             original_normal->vector.x, original_normal->vector.y, original_normal->vector.z);
//     }
//     catch(const std::exception& e)
//     {
//         ROS_WARN("Failed to transform normal: %s", e.what());
//     }
    
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "center2odom_node");
    ros::NodeHandle nh;

    tf2_ros::TransformListener tflistener(tfBuffer);

    // publish transformed center and normal
    transformed_center_pub = nh.advertise<geometry_msgs::PoseArray>("stair_center_odom", 1);
    // transformed_normal_pub = nh.advertise<geometry_msgs::Vector3Stamped>("stair_normal_odom", 1);

    // subscribe to original center and normal
    ros::Subscriber original_center_sub = nh.subscribe<geometry_msgs::PointStamped>("/stair_center", 1, centerCallback);
    // ros::Subscriber original_normal_sub = nh.subscribe<geometry_msgs::Vector3Stamped>("/stair_normal", 1, normalCallback);

    ros::spin();

    return 0;
}
