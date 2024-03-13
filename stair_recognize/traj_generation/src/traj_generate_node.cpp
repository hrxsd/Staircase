#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <algorithm>

class TrajectoryGenerator {
public:
    TrajectoryGenerator(): nh("~") {
        // 订阅三个台阶的中心点
        pose_array_sub = nh.subscribe("/stair_center_odom", 10, &TrajectoryGenerator::poseArrayCallback, this);

        // 发布轨迹
        path_pub = nh.advertise<nav_msgs::Path>("/trajectory", 1);

        // 获取机器人当前位置
        nh.param("robot_base_frame", robot_base_frame, std::string("base"));

        // 其他参数
        nh.param("interpolation_resolution", interpolation_resolution, 0.1);
        nh.param("z_offset", z_offset, 0.3);
        nh.param("interpolation_method", interpolation_method, std::string("bezier"));

        ROS_INFO("Trajectory Generator Initialized.");
    }

    void poseArrayCallback(const geometry_msgs::PoseArrayConstPtr& msg)
    {
        std::vector<geometry_msgs::Point> points;

        // 将机器人当前位置作为起点
        points.push_back(getCurrentRobotBasePosition());

        for (const auto& pose : msg->poses)
        {
            geometry_msgs::Point point = pose.position;
            // ROS_INFO("Original Point: x=%f, y=%f, z=%f", point.x, point.y, point.z);
            point.z = point.z + z_offset;
            // ROS_INFO("Modified Point: x=%f, y=%f, z=%f", point.x, point.y, point.z);
            points.push_back(point);
        }

        ROS_INFO("Received %lu points:", points.size());
        for (const auto& point : points)
        {
            ROS_INFO("Point: x=%f, y=%f, z=%f", point.x, point.y, point.z);
        }

        // 按照从近到远的顺序排列
        std::sort(points.begin(), points.end(), [](const geometry_msgs::Point& a, const geometry_msgs::Point& b) {
            return a.x < b.x;
        });

        // 根据插值方法进行插值
        nav_msgs::Path path;
        if (interpolation_method == "linear")
        {
            path = generateLinearPath(points);
        }
        else if (interpolation_method == "bezier")
        {
            path = generateBezierPath(points);
        }
        else if (interpolation_method == "cubic")
        {
            path = generateCubicPath(points);
        }
        else
        {
            ROS_WARN("Unknown interpolation method: %s", interpolation_method.c_str());
            path = generateLinearPath(points);
        }

        // nav_msgs::Path path = generatePath(points);

        path_pub.publish(path);
    }

    nav_msgs::Path generateLinearPath(const std::vector<geometry_msgs::Point>& points)
    {
        nav_msgs::Path path;
        path.header.frame_id = "odom";

        if (points.size() < 2)
        {
            ROS_WARN("Insufficient points for interpolation");
            return path;
        }

        for (size_t i=1; i<points.size(); ++i)
        {
            double distance = calculateDistance(points[i-1], points[i]);
            int num_steps = std::max(1, static_cast<int>(distance / interpolation_resolution));

            for (int step=0; step<=num_steps; ++step)
            {
                double t = static_cast<double>(step) / num_steps;
                geometry_msgs::PoseStamped pose_stamped;
                pose_stamped.header = path.header;
                pose_stamped.pose.position = interpolateLinear(points[i-1], points[i], t);
                path.poses.push_back(pose_stamped);
            }
        }

        return path;
    }

    nav_msgs::Path generateBezierPath(const std::vector<geometry_msgs::Point>& points)
    {
        nav_msgs::Path path;
        path.header.frame_id = "odom";

        if (points.size() < 2)
        {
            ROS_WARN("Insufficient points for interpolation");
            return path;
        }

        const int num_samples = 100;

        for (int i = 0; i < num_samples; ++i)
        {
            double t = static_cast<double>(i) / (num_samples - 1);

            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header = path.header;
            pose_stamped.pose.position = interpolateBezier(points, t);
            path.poses.push_back(pose_stamped);
        }

        return path;
    }
    // {
    //     nav_msgs::Path path;
    //     path.header.frame_id = "odom";

    //     if (points.size() < 2)
    //     {
    //         ROS_WARN("Insufficient points for interpolation");
    //         return path;
    //     }

    //     for (size_t i = 1; i < points.size(); ++i)
    //     {
    //         for (double t = 0.0; t <= 1.0; t += interpolation_resolution)
    //         {
    //             geometry_msgs::PoseStamped pose_stamped;
    //             pose_stamped.header = path.header;
    //             pose_stamped.pose.position = interpolateBezier(points[i-1], points[i], t);
    //             path.poses.push_back(pose_stamped);
    //         }
    //     }

    //     return path;
    // }

    nav_msgs::Path generateCubicPath(const std::vector<geometry_msgs::Point>& points)
    {
        nav_msgs::Path path;
        path.header.frame_id = "odom";

        if (points.size() < 2)
        {
            ROS_WARN("Insufficient points for interpolation");
            return path;
        }
        // 机器人当前位置作为起点
        // geometry_msgs::PoseStamped start_pose;
        // start_pose.header = path.header;
        // start_pose.pose.position = getCurrentRobotBasePosition();
        // path.poses.push_back(start_pose);

        // 生成轨迹
        for (size_t i = 1; i < points.size(); ++i)
        {
            for (double t = 0.0; t <= 1.0; t += interpolation_resolution)
            {
                geometry_msgs::PoseStamped pose_stamped;
                pose_stamped.header = path.header;
                pose_stamped.pose.position = interpolateCubic(points[i - 1], points[i], t);
                path.poses.push_back(pose_stamped);
            }
        }
        // for (double t = 0.0; t <= 1.0; t += interpolation_resolution)
        // {
        //     geometry_msgs::PoseStamped pose_stamped;
        //     pose_stamped.header = path.header;
        //     pose_stamped.pose.position = interpolate(points, t);
        //     path.poses.push_back(pose_stamped);
        // }

        return path;
    }

    geometry_msgs::Point interpolateLinear(const geometry_msgs::Point& point1, const geometry_msgs::Point& point2, double t)
    {
        geometry_msgs::Point interpolated_point;
        interpolated_point.x = (1.0 - t) * point1.x + t * point2.x;
        interpolated_point.y = (1.0 - t) * point1.y + t * point2.y;
        interpolated_point.z = (1.0 - t) * point1.z + t * point2.z;
        return interpolated_point;
    }

    geometry_msgs::Point interpolateCubic(const geometry_msgs::Point& point1, const geometry_msgs::Point& point2, double t)
    {
        // 三次多项式插值
        // if (points.size() < 4 || idx1 >= points.size() || idx2 >= points.size())
        // {
        //     ROS_WARN("Insufficient points for interpolation");
        //     return geometry_msgs::Point();
        // }

        double t2 = t*t;
        double t3 = t2*t;

        double h1 = 2*t3 - 3*t2 + 1;
        double h2 = -2*t3 + 3*t2;
        double h3 = t3 - 2*t2 + t;
        double h4 = t3 - t2;

        double x = h1*point1.x + h2*point2.x + h3*(point2.x - point1.x) + h4*(point2.x - point1.x);
        double y = h1*point1.y + h2*point2.y + h3*(point2.y - point1.y) + h4*(point2.y - point1.y);
        double z = h1*point1.z + h2*point2.z + h3*(point2.z - point1.z) + h4*(point2.z - point1.z);

        // 加入机器人当前位置
        x += robot_base_x;
        y += robot_base_y;
        z += robot_base_z;

        geometry_msgs::Point interpolated_point;
        interpolated_point.x = x;
        interpolated_point.y = y;
        interpolated_point.z = z;

        return interpolated_point;
    }

    geometry_msgs::Point interpolateBezier(const std::vector<geometry_msgs::Point>& points, double t)
    {
        // 贝塞尔曲线插值
        int n = points.size() - 1;

        geometry_msgs::Point interpolated_point;
        interpolated_point.x = 0.0;
        interpolated_point.y = 0.0;
        interpolated_point.z = 0.0;

        for (int i = 0; i <= n; ++i)
        {
            double coefficient = binomialCoefficient(n, i) * std::pow(1-t, n-i) * std::pow(t, i);
            interpolated_point.x += coefficient * points[i].x;
            interpolated_point.y += coefficient * points[i].y;
            interpolated_point.z += coefficient * points[i].z;
        }

        return interpolated_point;
    }
    // {
    //     double t2 = t*t;
    //     double t3 = t2*t;

    //     double b0 = 0.5*(-t3 + 2*t2 - t);
    //     double b1 = 0.5*(3*t3 - 5*t2 + 2);
    //     double b2 = 0.5*(-3*t3 + 4*t2 + t);
    //     double b3 = 0.5*(t3 - t2);

    //     double x = 0.5 * (point1.x*b0 +point2.x*b1);
    //     double y = 0.5 * (point1.y*b0 +point2.y*b1);
    //     double z = 0.5 * (point1.z*b0 +point2.z*b1);

    //     // 加入机器人当前位置
    //     x += robot_base_x;
    //     y += robot_base_y;
    //     z += robot_base_z;

    //     geometry_msgs::Point interpolated_point;
    //     interpolated_point.x = x;
    //     interpolated_point.y = y;
    //     interpolated_point.z = z;

    //     return interpolated_point;
    // }

    double calculateDistance(const geometry_msgs::Point& point1, const geometry_msgs::Point& point2)
    {
        // 计算两点之间的欧式距离
        double dx = point1.x - point2.x;
        double dy = point1.y - point2.y;
        double dz = point1.z - point2.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    geometry_msgs::Point getCurrentRobotBasePosition()
    {
        geometry_msgs::TransformStamped transformStamped;
        try 
        {
            transformStamped = tf_buffer.lookupTransform("odom", robot_base_frame, ros::Time(0));
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN("%s", ex.what());
            return geometry_msgs::Point();
        }

        // 将vector3转换为point
        geometry_msgs::Point current_robot_base_position;
        current_robot_base_position.x = transformStamped.transform.translation.x;
        current_robot_base_position.y = transformStamped.transform.translation.y;
        current_robot_base_position.z = transformStamped.transform.translation.z;
        
        return current_robot_base_position;
    }

    int binomialCoefficient(int n, int k)
    {
        // 计算二项式系数
        if (k == 0 || k == n)
        {
            return 1;
        }
        else
        {
            int result = 1;
            for (int i = 1; i <= k; ++i)
            {
                result *= (n - i + 1);
                result /= i;
            }
            return result;
        }
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber pose_array_sub;
    ros::Publisher path_pub;

    std::string robot_base_frame;
    double robot_base_x, robot_base_y, robot_base_z;
    double interpolation_resolution;
    double z_offset;
    std::string interpolation_method;

    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener{tf_buffer};
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "traj_generate_node");

    TrajectoryGenerator generator;

    ros::spin();

    return 0;
}
