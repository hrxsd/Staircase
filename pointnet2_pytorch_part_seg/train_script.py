import sys
sys.path.append(r"/home/robota1003a/pointnet2_pytorch_part_seg")
import train 
import argparse
from data_utils.data_recorder import *
import numpy as np
import random

# hyperparameters
EPOCH_LIST          = [200,250,300]#一般越多越好
NPOINT_LIST         = [2048,4096] #会不会跟算法向量的部分有联系
BATCHSIZE_LIST      = [16,32,64] #跟其他无关，单独训练

RRNS_PROBILITY_LIST = [0.2,0.3,0.4,0.5,0.6,0.7,0.8] # 立板去掉的概率 基本0.5最佳 因为一半这种工业楼梯很合理，但不知道影不影响其他。
RRB_RADIUS_LIST     = [0.2,0.5,0.8] # 这个有个最优值。0.1
RGM_STEP_LIST       = [0.001,0.0015,0.002,0.003,0.004,0.006,0.008,0.01] #越大越不好，但是有最优值。这个单独来0.002
LOSS_K_LIST         = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #这个估计越做越不好，但是有最优值。，这个单独来0.1
LOSS_RADIUS_LIST    = [0.05,0.1,0.2] #上次实验得知，radius越大越好。跟npoint也相关


def parse_args():
    parser = argparse.ArgumentParser('Model') #生成parser对象的方法
    parser.add_argument('--model',  type=str,           default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--learning_rate',type=float,   default=0.001,              help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str,              default='0',                help='GPU to use [default: GPU 0]')
    #parser.add_argument('--optimizer', type=str,        default='Adam',             help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str,          default=None,               help='Log path [default: None]') #可以指定log的文件夹名
    parser.add_argument('--decay_rate', type=float,     default=1e-4,               help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,        default=20,                 help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,       default=0.7,                help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int,        default=5,                  help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--trian_dataset',type=str,     default='/home/robota1003a/pointnet2_pytorch_part_seg/dataset/dataset_withlabell_trian_pointnormal',help='train dataset root path')
    parser.add_argument('--eval_dataset',type=str,      default='/home/robota1003a/pointnet2_pytorch_part_seg/dataset/dataset_withlabell_trian_pointnormal',help='eval dataset root path')

    parser.add_argument('--epoch', type=int,            default=200,                help='Epoch to run [default: 32]')
    parser.add_argument('--npoint', type=int,           default=80000,               help='Point Number [default: 4096]')
    parser.add_argument('--batch_size', type=int,       default=16,                 help='Batch Size during training [default: 16]')
    parser.add_argument('--rrns_prob',type=float,       default=0.5,             help='radius for caculating norms of loss')
    parser.add_argument('--rrb_radius',type=float,      default=0.1,             help='radius for caculating norms of loss')
    parser.add_argument('--rgm_step',type=float,        default=0.002,            help='radius for caculating norms of loss')
    parser.add_argument('--loss_k',type=float,          default=0.1,              help='loss互补系数')
    parser.add_argument('--loss_radius',type=float,     default=0.03,             help='radius for caculating norms of loss')
    
    parser.add_argument('--newloss_enable',type=bool, default=True,               help='newloss_enable')

    parser.add_argument('--rrns_enable',type=bool, default=True,               help='random remove non steps')
    parser.add_argument('--rrb_enable',type=bool,default=True,                 help='random remove ball')
    parser.add_argument('--rrs_enable',type=bool, default=True,                help='random remove steps')
    parser.add_argument('--rgm_enable',type=bool, default=True,                help='random gaussion move')
    parser.add_argument('--rgrp_enable',type=bool,default=True,                help='random gaussion remove point')
    parser.add_argument('--rspo_eanble',type=bool, default=True,               help='random scramble point order')
   
    return parser.parse_args()
if __name__ == '__main__':

    order = 0
    for npoint in NPOINT_LIST:
        for rrb_radius in RRB_RADIUS_LIST:
            for loss_radius in LOSS_RADIUS_LIST:
                #data recorder init
                experiment = 'finetune'
                data_record_path = '/home/robota1003a/pointnet2_pytorch_part_seg/experiment'
                dr = data_recorder(experiment_name=experiment,write_path=data_record_path,files_order=order)
                args = parse_args()
                args.epoch = 200
                args.npoint = npoint
                args.batch_size = 10
                args.rrns_prob = 0.5
                args.rrb_radius = rrb_radius
                args.rgm_step = 0.002
                args.loss_k = 0.1
                args.loss_radius = loss_radius
                args.newloss_enable = True
                args.rrns_enable = True
                args.rrb_enable  = True
                args.rrs_enable  = True
                args.rgm_enable  = True
                args.rgrp_enable = True
                args.rspo_eanble = True
                train.main(args,dr,order)
                order += 1