#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

ros::Publisher marker_pub;

void visCenterCallback(const geometry_msgs::PointStamped::ConstPtr& transformed_center)
{
    // 创建一个中心点的marker
    visualization_msgs::Marker point_marker;
    point_marker.header.frame_id = "odom";
    point_marker.header.stamp = ros::Time();
    point_marker.ns = "points";
    point_marker.id = 0;
    point_marker.type = visualization_msgs::Marker::SPHERE;
    point_marker.action = visualization_msgs::Marker::ADD;

    //设置点的位置
    point_marker.pose.position.x = transformed_center->point.x;
    point_marker.pose.position.y = transformed_center->point.y;
    point_marker.pose.position.z = transformed_center->point.z;
    point_marker.pose.orientation.x = 0.0;
    point_marker.pose.orientation.y = 0.0;
    point_marker.pose.orientation.z = 0.0;
    point_marker.pose.orientation.w = 1.0;

    //点的尺寸和颜色
    point_marker.scale.x = 0.1;
    point_marker.scale.y = 0.1;
    point_marker.scale.z = 0.1;
    point_marker.color.r = 1.0f;
    point_marker.color.g = 0.0f;
    point_marker.color.b = 0.0f;
    point_marker.color.a = 1.0;
    point_marker.lifetime = ros::Duration(1.0);

    // 发布marker
    marker_pub.publish(point_marker);
}

void visNormalCallback(const geometry_msgs::Vector3Stamped::ConstPtr& transformed_normal)
{
    // 创建一个法向量的marker
    visualization_msgs::Marker normal_marker;
    normal_marker.header.frame_id = "odom";
    normal_marker.header.stamp = ros::Time();
    normal_marker.ns = "normals";
    normal_marker.id = 0;
    normal_marker.type = visualization_msgs::Marker::ARROW;
    normal_marker.action = visualization_msgs::Marker::ADD;

    //设置法向量的起点
    normal_marker.points.resize(2);
    normal_marker.points[0].x = 0.0;
    normal_marker.points[0].y = 0.0;
    normal_marker.points[0].z = 0.0;

    //设置法向量的终点
    normal_marker.points[1].x = transformed_normal->vector.x;
    normal_marker.points[1].y = transformed_normal->vector.y;
    normal_marker.points[1].z = transformed_normal->vector.z;

    //法向量的尺寸和颜色
    normal_marker.scale.x = 0.05;
    normal_marker.scale.y = 0.1;
    normal_marker.scale.z = 0.1;
    normal_marker.color.r = 0.0f;
    normal_marker.color.g = 0.0f;
    normal_marker.color.b = 1.0f;
    normal_marker.color.a = 1.0;
    normal_marker.lifetime = ros::Duration(1.0);

    //发布marker
    marker_pub.publish(normal_marker);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "visualization_node");
    ros::NodeHandle nh;

    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    ros::Subscriber transformed_center_sub = nh.subscribe<geometry_msgs::PointStamped>("stair_center_odom", 1, visCenterCallback);
    ros::Subscriber transformed_normal_sub = nh.subscribe<geometry_msgs::Vector3Stamped>("stair_normal_odom", 1, visNormalCallback);

    ros::spin();

    return 0;
}
