import pandas as pd
#import openpyxl
import datetime
import os

####################################################
#
#
#
####################################################
class data_recorder():
    def __init__(self,experiment_name,write_path): #初始化数据记录样板
        self.experiment_data = None #保存的实验数据的类型

        #初始化文件保存名
        self.write_path = write_path # 存储路径
        self.current_datetime = datetime.datetime.now()
        self.current_date = self.current_datetime.date()
        self.current_time = self.current_datetime.time()

        if experiment_name == 'time':
            self.experiment_data = {
            '日期':[],
            '版本':[],
            '数据集':[],
            'index':[],
            '网络预测时间':[],
            '平面计算时间':[],
            '中心点计算时间':[],
            '参数计算时间':[]
            }
            self.file_name = 'timecost_record_'+str(self.current_date)+str(self.current_time)+'.xlsx'
            
        if experiment_name == 'normal':
            #实验记录用数据结构
            self.experiment_data = {
            '日期':[],
            '版本':[],
            '数据集':[],
            'index':[],
            '预测法向量':[],
            'GT法向量':[],
            '余弦距离误差':[]
            }
            self.file_name = 'normal_error_record_'+str(self.current_date)+str(self.current_time)+'.xlsx'
        
        if experiment_name == 'finetune':
            #实验记录用数据结构
            self.experiment_data = {
            '日期':[],
            '训练集':[],
            '验证集':[],
            'epoch':[],
            'npoint':[],
            'batch_size':[],
            'rrns_prob':[],
            'rrb_radius':[],
            'rgm_step':[],
            'loss_k':[],
            'loss_radius':[],
            'training_Accuracy':[],
            'eval_Accuracy':[],
            'mIoU':[],
            'best_mIoU':[],
            'training_Loss':[],
            'eval_Loss':[]
            }
            self.file_name = 'finetune_'+str(self.current_date)+str(self.current_time)+'.xlsx'
        
        if experiment_name == 'test': #debug 
            self.experiment_data = {
            '1':[],
            '2':[]
            }
            self.file_name = 'test_data_recorder_'+str(self.current_date)+str(self.current_time)+'.xlsx'
        self.full_path = os.path.join(self.write_path,self.file_name)
    #单条数据记录器
    def record_data(self,key,value):
        self.experiment_data[key].append(value)
    #全部实验记录之后压入Excel
    def record2excel(self):
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(self.experiment_data)
        # 将数据写入Excel文件
        self.df.to_excel(self.full_path, index=False)





#调试成功 2023年6月26日
if __name__ == '__main__':
    # 示例数据
    dr = data_recorder('test',r'D:\\Project_on_going\\pointnet2_pytorch_part_seg')
    dr.record_data('1','one')
    dr.record_data('2','two')
    dr.record2excel()
