"""
Author: Benny
Date: Nov 2019
"""
import os
import sys
sys.path.append(r"D:\\Project_on_going\\pointnet2_pytorch_part_seg")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
from models.provider import random_point_dropout,random_scale_point_cloud,shift_point_cloud,jitter_point_cloud,rotate_point_cloud_z
import numpy as np
import time
'''修改 2023年1月24日12点25分'''
from data_utils.trainning_dataset import Train_DataSet,Eval_DataSet
# 调试用
#np.set_printoptions(threshold=np.inf) #numpy数据打印长度问题
from train_script import *

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args,data_recorder,order):

    #__file__表示显示文件当前的位置 但是：如果当前文件包含在sys.path里面，那么，__file__返回一个相对路径！如果当前文件不包含在sys.path里面，那么__file__返回一个绝对路径！
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    #场景类别
    classes = ['step0','step1','step2','step3','step4','step5','step6','step7','step8','step9','notsteps']
    class2label = {cls: i for i, cls in enumerate(classes)} #{'step0': 0, 'step1': 1, 'step2': 2, 'step3': 3, 'step4': 4, 'step5': 5, 'step6': 6, 'step7': 7, 'step8': 8, 'step9': 9}
    print(class2label)

    #类别编码
    seg_classes = class2label
    seg_label_to_cat = {} #label 和 category 列表
    for i, cat in enumerate(seg_classes.keys()): #长这样子：i,cat :   (0,'step0'), (1, 'step3'), (2, 'step2')
        seg_label_to_cat[i] = cat

    #日志输出控制台
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #日志文件夹组成:./log/sem_seg/timestr/checkpoints|/logs
    #timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    order_str = str(order)
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('experiment')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(order_str)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    #日志对象设置
    #args = parse_args() #调用argserver
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#定义handler的输出格式（formatter）
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))#创建一个handler，用于写入日志文件
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #数据集路径和数据集对象生成加载
    TRAIN_DATASET_PATH = args.trian_dataset
    EVAL_DATASET_PATH  = args.eval_dataset
    NUM_CLASSES = 11

    #模型加载和当前配置转存到log文件夹
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))#Python标准库,作为os模块补充，提供复制、移动、删除、压缩、解压等操作,这些 os 模块中一般是没有提供的。
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    shutil.copy('data_utils/trainning_dataset.py',str(experiment_dir))
    shutil.copy('train.py',str(experiment_dir))
    
    #分类器生成
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion_new  = MODEL.get_loss_new(args.loss_radius,args.loss_k).cuda()
    criterion      = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    #参数初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    #模型参数加载
    #现有模型的标志位
    #experiment_dir = r'D:\\Project_on_going\\pointnet2_pytorch_part_seg\\log\sem_seg\\2023-07-28_12-30'
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    
    #优化器
    optimizer = torch.optim.Adam(classifier.parameters(),lr=args.learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=args.decay_rate)

    #调整优化器动量
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5 #最小学习率 0.00001
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0 #这个是本轮具体运行了多少epoch
    best_iou = 0
    # start epoch 可以根据预加载模型的参数进行调整

    #随机点数调整
    #NPOINT_LIST = [1024,2048,4096,8192] #会不会跟算法向量的部分有联系
    NUM_POINT = args.npoint

    BATCH_SIZE = args.batch_size
    TRAIN_DATASET = Train_DataSet(root = TRAIN_DATASET_PATH,
                                  npoints = NUM_POINT,
                                  rrns_prob = args.rrns_prob,
                                  rrb_radius_max = args.rrb_radius,
                                  rgm_step_divetion = args.rgm_step,
                                  rrns_enable = args.rrns_enable,
                                  rrb_enable = args.rrb_enable,
                                  rrs_enable = args.rrs_enable,
                                  rgm_enable = args.rgm_enable,
                                  rgrp_enable = args.rgrp_enable,
                                  rspo_eanble = args.rspo_eanble)
    TEST_DATASET  = Eval_DataSet(root = EVAL_DATASET_PATH, npoints=NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=True, drop_last=True,worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader  = torch.utils.data.DataLoader(TEST_DATASET,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=True, drop_last=True)
    print("The number of training data and test data is: %d" %len(TRAIN_DATASET))
    print("Current npoint selection is:",NUM_POINT)

    for epoch in range(start_epoch, args.epoch):
        
        # # 随机点数调整
        # #NPOINT_LIST = [1024,2048,4096,8192] #会不会跟算法向量的部分有联系
        # #NUM_POINT = NPOINT_LIST[random.randint(0,3)]
        # NUM_POINT = 2048
        # BATCH_SIZE = args.batch_size
        # TRAIN_DATASET = MyDataSet(root = TRAIN_DATASET_PATH,npoints=NUM_POINT,rrns_prob=args.rrns_prob,rrb_radius_max = args.rrb_radius,rgm_step_divetion=args.rgm_step)
        # TEST_DATASET  = MyDataSet(root = EVAL_DATASET_PATH, npoints=NUM_POINT,rrns_prob=args.rrns_prob,rrb_radius_max = args.rrb_radius,rgm_step_divetion=args.rgm_step)
    
        # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,pin_memory=True, drop_last=True,worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
        # testDataLoader  = torch.utils.data.DataLoader(TEST_DATASET,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=True, drop_last=True)
        # print("The number of training data and test data is: %d" %len(TRAIN_DATASET))
        # print("Current npoint selection is:",NUM_POINT)

        log_string('---- Epoch %d (%d/%s)TRAIN ----' %(global_epoch + 1, epoch + 1, args.epoch))

        # 学习率调整
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Current Learning Rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # BN momentum 调整
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        
        # batch 训练
        for i, (points,points_gt_normal,target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            
            #清空过往梯度
            optimizer.zero_grad()
            
            #转numpy
            points = points.data.numpy() 
            points_gt_normal = points_gt_normal.data.numpy()
            #数据增强
            points[:, :, :3], points_gt_normal[:, :, :3] = random_point_dropout(points[:, :, :3], points_gt_normal[:, :, :3]) #这里删掉一些点之后怎么保证跟target一样维度，没删，只是变成了一号点的位置
            points[:, :, :3], points_gt_normal[:, :, :3] = rotate_point_cloud_z(points[:, :, :3], points_gt_normal[:, :, :3])
            points[:, :, :3] = random_scale_point_cloud(points[:, :, :3])
            points[:, :, :3] =        shift_point_cloud(points[:, :, :3])
            #转tensor
            points = torch.Tensor(points)#numpy转换为tensor的函数，但不共享内存，转换较慢

            #转cuda
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1) #B*N*C 变 B*C*N
            
            #预测
            seg_pred, trans_feat = classifier(points)#seg_pred 大小为 32*2048*11
            #new loss needs
            seg_pred_has_batch = seg_pred
            target_has_batch = target
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) #整形 65536*11 大小

            #目标值转65536*1，取全部放CPU里，转numpy；32*2048 = 65536
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            #目标值转65536*1
            target = target.view(-1, 1)[:, 0]
            
            # 32*S*N'*3的点云
            # epoch 超过50代之后加入newLoss算法
            if epoch >= 50 and args.newloss_enable == True:
                #Input: seg_pred:               Tensor in GPU, 65536*11
                #       seg_pred_has_batch:     Tensor in GPU, 32*2048*11
                #       target:                 Tensor in GPU, 65536*1 
                #       target_has_batch:       Tensor in GPU, 32*2048*1
                #       points:                 List , 32*3*2048
                #       points_gt_normal:       List , 32*2048*3
                loss = criterion_new(seg_pred, seg_pred_has_batch, target, target_has_batch, points, points_gt_normal) #trans_feat)#,weights)
            else:
                loss = criterion(seg_pred,target)#,trans_feat)

            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        
        training_mean_loss = loss_sum / num_batches
        log_string('Training mean loss: %f' % (training_mean_loss))
        training_accuracy = total_correct / float(total_seen)
        log_string('Training accuracy: %f' % (training_accuracy))
        
        # 每5次保存一次
        if epoch % 5 == 0:
            print("Save model")
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # 验证
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class     = [0 for _ in range(NUM_CLASSES)]
            total_correct_class  = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points,points_gt_normal,target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                #qc add
                seg_pred_has_batch = seg_pred
                target_has_batch = target
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                #loss = criterion(seg_pred, target, trans_feat)#, weights)
                #qc add
                # 32*S*N'*3的点云
                # epoch 超过50代之后加入newLoss算法
                if epoch >= 50 and args.newloss_enable == True:
                    #Input: seg_pred:               Tensor in GPU, 65536*11
                    #       seg_pred_has_batch:     Tensor in GPU, 32*2048*11
                    #       target:                 Tensor in GPU, 65536*1 
                    #       target_has_batch:       Tensor in GPU, 32*2048*1
                    #       points:                 List , 32*3*2048
                    #       points_gt_normal:       List , 32*2048*3
                    loss = criterion_new(seg_pred, seg_pred_has_batch, target, target_has_batch, points, points_gt_normal) #trans_feat)#,weights)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (BATCH_SIZE * NUM_POINT)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp
                else:
                    loss = criterion(seg_pred,target)#,trans_feat)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (BATCH_SIZE * NUM_POINT)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            eval_mean_loss = loss_sum / num_batches
            log_string('Eval mean loss: %f' % (eval_mean_loss))
            eval_accuracy = total_correct / float(total_seen)
            log_string('Eval accuracy: %f' % (eval_accuracy))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
            log_string('\n')
        global_epoch += 1

        # 数据记录
        data_recorder.record_data('日期',datetime.datetime.now().date())
        data_recorder.record_data('训练集',TRAIN_DATASET_PATH)
        data_recorder.record_data('验证集',EVAL_DATASET_PATH)
        data_recorder.record_data('epoch',epoch)
        data_recorder.record_data('npoint',NUM_POINT)
        data_recorder.record_data('batch_size',BATCH_SIZE)
        data_recorder.record_data('rrns_prob',args.rrns_prob)
        data_recorder.record_data('rrb_radius',args.rrb_radius)
        data_recorder.record_data('rgm_step',args.rgm_step)
        data_recorder.record_data('loss_k',args.loss_k)
        data_recorder.record_data('loss_radius',args.loss_radius)
        data_recorder.record_data('training_Accuracy',training_accuracy)
        data_recorder.record_data('eval_Accuracy',eval_accuracy)
        data_recorder.record_data('mIoU',mIoU)
        data_recorder.record_data('best_mIoU',best_iou)
        data_recorder.record_data('training_Loss',training_mean_loss.cpu().detach().numpy())
        data_recorder.record_data('eval_Loss',eval_mean_loss.cpu().detach().numpy())   
    data_recorder.record2excel()


if __name__ == '__main__':
    
    #data recorder init
    experiment = 'finetune'
    data_record_path = 'D:\Project_on_going\pointnet2_pytorch_part_seg\log\sem_seg\experiment'
    dr = data_recorder(experiment_name=experiment,write_path=data_record_path)
    
    args = parse_args()
    main(args,dr)
    
