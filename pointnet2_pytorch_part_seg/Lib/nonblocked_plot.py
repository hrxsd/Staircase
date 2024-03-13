# 必须
import numpy as np
import matplotlib.pyplot as plt

# 测试功能
import random 


# import matplotlib.pyplot as plt

# class MyPlotter:
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()  # 创建图形和坐标轴
#         self.ax.ion()
#         self.show()
#         self.ax.set_xlabel('X')
#         self.ax.set_ylabel('Y')
#         self.ax.set_title('Custom Plot')
#     def custom_plot(self, x, y):
#         self.ax.plot(x, y, 'r-')

#     def show(self):
#         plt.show()

# # 创建自定义绘图类的实例
# plotter = MyPlotter()
# plotter2 = MyPlotter()
# # 使用自定义绘图方法绘制图形
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
# plotter.custom_plot(x, y)

# # 显示图形
# plotter.show()
# plotter2.show()
# class Myplotter(plt):
#     def __init__(self,epoch,name) -> None:
#         super().__init__() 
#         self.title(name)
#         self.xlim(0,epoch)
#         self.ylim(0,2)
#         self.ion()
#         self.show()
#         self.x = []
#         self.y = []
#     def update(self): 
#         self.plot(self.x,self.y)
#         self.draw()
#         self.pause(0.001)

# if __name__ == '__main__':
#     pltr = Myplotter(200,'name')
#     pltr1 = Myplotter(200,'qc')
#     #x = np.arange(0,200)
#     for i in range(0,200):
#         pltr.x.append(i)
#         pltr.y.append(random.uniform(0, 2))
#         pltr.update()
#         pltr1.x.append(i)
#         pltr1.y.append(random.uniform(0, 2))
#         pltr1.update()

# import pyformulas as pf
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# fig = plt.figure()

# canvas = np.zeros((480,640))
# screen = pf.screen(canvas, 'Sinusoid')

# start = time.time()
# while True:
#     now = time.time() - start

#     x = np.linspace(now-2, now, 100)
#     y = np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
#     plt.xlim(now-2,now+1)
#     plt.ylim(-3,3)
#     plt.plot(x, y, c='black')

#     # If we haven't already shown or saved the plot, then we need to draw the figure first...
#     fig.canvas.draw()

#     image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     screen.update(image)

# #screen.close()