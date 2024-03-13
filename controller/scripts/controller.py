#!/usr/bin/env python
import numpy as np
import sys
import rospy
import tf
import math
import sys
import select
import os
if os.name == 'nt':
    import msvcrt
else:
    import tty
    import termios
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Float32, Int16
import matplotlib.pyplot as plt
import threading
from scipy import interpolate
from scipy.interpolate import CubicSpline


class Controller():
    def __init__(self):
        self.N = 6
        
        self.Kp_x = 2.0
        self.Ki_x = 0.0
        self.Kd_x = 0.0
        
        self.Kp_y = 0.7
        self.Ki_y = 0.0
        self.Kd_y = 0.0
        
        # 保存历史参考输入值的列表
        self.ref_inputs_history = []
        self.filter_window_size = 5
        
        # 初始化一个缓冲区和计数器
        self.buffer_size = 10  # 用于保存状态的缓冲区大小
        self.state_buffer = []  # 用于保存状态的缓冲区
        self.update_frequency = 1 # 更新频率
        self.update_counter = 0 # 更新计数器
        
        # 用于保存样条函数
        # self.spline_func = None
        
        # 用于可视化
        self.control_signal_x_list = []
        self.ref_inputs_list = []
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.set_window_title('Control Signal X vs Reference Inputs')
                
        self.rate = rospy.Rate(50)
        self.curr_state = np.zeros(4)
        self.sub1 = rospy.Subscriber(
            '/local_plan', Float32MultiArray, self.local_planner_cb)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub2 = rospy.Publisher(
            '/curr_state', Float32MultiArray, queue_size=10)
        self.__timer_localization = rospy.Timer(
            rospy.Duration(0.01), self.get_current_state)
        self.listener = tf.TransformListener()
        self.have_plan = 0
        self.curr_time = 0
        self.time_sol = 0
        self.local_plan = np.zeros([self.N, 2])
        self.control_cmd = Twist()
        self.control_loop()



    def quart_to_rpy(self, x, y, z, w):
        r = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        p = math.asin(2*(w*y-z*x))
        y = math.atan2(2*(w*z+x*y), 1-2*(z*z+y*y))
        return r, p, y

    def get_current_state(self, event):
        try:
            (trans, rot) = self.listener.lookupTransform(
                'odom', 'base', rospy.Time(0))

            # # 将状态信息保存到缓冲区中
            # self.state_buffer.append(np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]]))
            # # 缓冲区达到指定大小时，更新状态
            # if len(self.state_buffer) >= self.buffer_size:
            #     if self.update_counter >= self.update_frequency:
            #         # 计算状态的平均值
            #         self.curr_state = np.mean(self.state_buffer, axis=0)
            #         self.state_buffer = []  # 清空缓冲区
            #         self.update_counter = 0 # 重置计数器
            #     else:
            #         self.update_counter += 1
            

            self.curr_state[0] = trans[0]
            self.curr_state[1] = trans[1]
            self.curr_state[2] = trans[2]
            roll, pitch, self.curr_state[3] = self.quart_to_rpy(
                rot[0], rot[1], rot[2], rot[3])  # r,p,y
                
            c = Float32MultiArray()
            c.data = [self.curr_state[0], self.curr_state[1], self.curr_state[2],
                    (self.curr_state[3]+np.pi) % (2*np.pi)-np.pi, roll, pitch]
            
            # c = Float32MultiArray()
            # c.data = [self.curr_state[0], self.curr_state[1], self.curr_state[2],
            #           (self.curr_state[3] + np.pi)% (2 * np.pi) - np.pi, 
            #           self.curr_state[4], self.curr_state[5], self.curr_state[6]]
            self.pub2.publish(c)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def cmd(self, data):
        self.control_cmd.linear.x = data[0]
        self.control_cmd.angular.z = data[1]
        print("control input: ", data)
        self.pub.publish(self.control_cmd)

    def getKey(self):
        if os.name == 'nt':
            return msvcrt.getch()

        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def control_loop(self):
        while not rospy.is_shutdown():
            start_auto = self.manual()
            if(start_auto):
                end_auto = self.auto()
                if not end_auto:
                    break
        self.cmd(np.array([0.0, 0.0]))

    def auto(self):
        prev_error_x = 0.0
        integral_x = 0.0
        
        prev_error_y = 0.0
        integral_y = 0.0
        
        smoothed_ref_inputs_x = 0.0
        
        plt.ion()
        
        while not rospy.is_shutdown():
            key = self.getKey()
            if key == 'q':
                return True
            
            ref_inputs = self.local_plan[5]
            current_position = self.curr_state[:2]

            error_x = smoothed_ref_inputs_x - current_position[0]
            integral_x += error_x
            derivative_x = error_x - prev_error_x
            control_signal_x = self.Kp_x * error_x + self.Ki_x * integral_x + self.Kd_x * derivative_x
            prev_error_x = error_x
            control_signal_x = np.clip(control_signal_x, -0.25, 0.25)
            
            error_y = ref_inputs[1] - current_position[1]
            integral_y += error_y
            derivative_y = error_y - prev_error_y
            control_signal_y = self.Kp_y * error_y + self.Ki_y * integral_y + self.Kd_y * derivative_y
            prev_error_y = error_y
            control_signal_y = np.clip(control_signal_y, -0.01, 0.01)
            
            # 将当前参考输入值添加到历史列表中
            self.ref_inputs_history.append(ref_inputs[0])
            
            # self.update_spline_fuction()
            
            # 滤波
            smoothed_ref_inputs_x = self.smopth_ref_inputs()
            
            # 可视化
            self.control_signal_x_list.append(control_signal_x)
            self.ref_inputs_list.append(error_x)
            
            # 估算当前参考输入的值
            # estimated_ref_inputs = self.estimate_ref_inputs()
            # self.ref_inputs_list.append(estimated_ref_inputs)
            
            self.cmd(np.array([control_signal_x, control_signal_y]))
            
            # self.update_graph()
            
            # 用于可视化
            self.plot_graph()
            self.rate.sleep()
        
        plt.ioff()
        plt.show()
        
    # def update_graph(self):
    #     self.line_control.set_xdata(range(len(self.control_signal_x_list)))
    #     self.line_control.set_ydata(self.control_signal_x_list)
    #     self.line_reference.set_xdata(range(len(self.ref_inputs_list)))
    #     self.line_reference.set_ydata(self.ref_inputs_list)

    #     # 重新绘制图表
    #     self.ax.relim()
    #     self.ax.autoscale_view()
    #     plt.draw()
    #     plt.pause(0.01)
        
    # def show_graph(self):
    #     plt.show()
    
    
    
    # 线性插值估算当前参考输入的值
    # def estimate_ref_inputs(self):
        # if len(self.ref_inputs_history) > 1:
        #     # 创建一个线性插值函数
        #     interp_func = interpolate.interp1d(
        #         np.arange(len(self.ref_inputs_history)), 
        #         self.ref_inputs_history, 
        #         kind='linear', 
        #         fill_value='extrapolate'
        #         )
            
        #     # 计算当前参考输入的值
        #     estimated_ref_inputs = interp_func(len(self.ref_inputs_history) - 1)
            
        #     return estimated_ref_inputs
        # else:
        #     return 0.0
    #     if self.spline_func is not None:
    #         if len(self.ref_inputs_history) > 1:
    #             estimated_ref_inputs = self.spline_func(len(self.ref_inputs_history) - 1)
    #             return estimated_ref_inputs
    #     return 0.0
    
    
    # def update_spline_fuction(self):
    #     if len(self.ref_inputs_history) > 1:
    #         t = np.arange(len(self.ref_inputs_history))
    #         self.spline_func = CubicSpline(t, self.ref_inputs_history)
    
    
    
    def smopth_ref_inputs(self):
        if len(self.ref_inputs_history) < self.filter_window_size:
            return self.ref_inputs_history[-1]
        smoothed_value = np.mean(self.ref_inputs_history[-self.filter_window_size:])
        return smoothed_value
    
    def plot_graph(self):
        self.ax.clear()
        
        # 绘制控制信号和期望位置信息的图表
        self.ax.plot(self.control_signal_x_list, label='Control Signal X')
        self.ax.plot(self.ref_inputs_list, label='error_x')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Control Signal X vs error_x')
        self.ax.legend()
        self.ax.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # 暂停一小段时间以允许图表更新

    def manual(self):
        data = np.array([0.0, 0.0])
        while not rospy.is_shutdown():
            key = self.getKey()
            if key == 'w':
                if(data[0] < 0.3):
                    data[0] += 0.05
                else:
                    data = data
            elif key == 'x':
                if(data[0] > -0.3):
                    data[0] -= 0.05
                else:
                    data = data
            elif key == 'a':
                if(data[1] < 0.6):
                    data[1] += 0.2
                else:
                    data = data
            elif key == 'd':
                if(data[1] > -0.6):
                    data[1] -= 0.2
                else:
                    data = data
            elif key == 'q':
                if(data[0] < 0.3):
                    data[0] += 0.1
                else:
                    data = data
                if(data[1] < 0.3):
                    data[1] += 0.1
                else:
                    data = data
            elif key == 'e':
                if(data[0] < 0.3):
                    data[0] += 0.1
                else:
                    data = data
                if(data[1] > -0.3):
                    data[1] -= 0.1
                else:
                    data = data
            elif key == 'c':
                if(data[0] > -0.3):
                    data[0] -= 0.1
                else:
                    data = data
                if(data[1] > -0.3):
                    data[1] -= 0.1
                else:
                    data = data
            elif key == 'z':
                if(data[0] > -0.3):
                    data[0] -= 0.1
                else:
                    data = data
                if(data[1] < 0.3):
                    data[1] += 0.1
                else:
                    data = data      
            elif key == 's':
                data = np.array([0.0, 0.0])
            elif key == 'i':
                return True
            elif (key == '\x03'):
                return False
            else:
                data = data
            self.cmd(data)
            self.rate.sleep()
            
    def local_planner_cb(self, msg):
        for i in range(self.N):
            self.local_plan[i, 0] = msg.data[0+2*i]
            self.local_plan[i, 1] = msg.data[1+2*i]


if __name__ == '__main__':
    rospy.init_node('control')
    controller = Controller()
