"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import sys
sys.path.append("D:\pytorch_project\pointnet2_pytorch-master")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
from models.provider import rotate_point_cloud_z
import numpy as np
import time
'''修改 2023年1月24日12点25分'''
from data_utils.mydataset import MyDataSet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#场景类别
classes = ['notstairs', 'stairs', 'flatarea']
class2label = {cls: i for i, cls in enumerate(classes)}
print(class2label)
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model') #生成parser对象的方法
    parser.add_argument('--model',  type=str,           default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int,       default=32,                 help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', type=int,            default=100,                help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate',type=float,   default=0.001,              help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str,              default='0',                help='GPU to use [default: GPU 0]')
    #parser.add_argument('--optimizer', type=str,        default='Adam',             help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str,          default=None,               help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float,     default=1e-4,               help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,           default=4000,               help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,        default=20,                 help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,       default=0.7,                help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int,        default=5,                  help='Which area to use for test, option: 1-6 [default: 5]')
    # 一共有6个area，其中第5用于测试
    return parser.parse_args()

def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'D:\pytorch_project\StairsSet\downsampled_label3' #################
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    TRAIN_DATASET = MyDataSet(root = root, npoints=NUM_POINT,label=2)
    TEST_DATASET  = MyDataSet(root = root, npoints=NUM_POINT,label=2)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,pin_memory=True, drop_last=True,worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader  = torch.utils.data.DataLoader(TEST_DATASET,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=True, drop_last=True)
    
    #weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    print("The number of training data and test data is: %d" %len(TRAIN_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion  = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    
    optimizer = torch.optim.Adam(classifier.parameters(),lr=args.learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=args.decay_rate)

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
    for epoch in range(start_epoch, args.epoch):
        print('---- Epoch %d (%d/%s) ----' %(global_epoch + 1, epoch + 1, args.epoch))

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        print('Current Learning Rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        #BN momentum 调整
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        #print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        
        # 每个batch batch 训练
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()#清空过往梯度；

            points = points.data.numpy() #转numpy数组
            #对所有点云数据增强
            points[:, :, :3] = rotate_point_cloud_z(points[:, :, :3])
            #numpy转换为tensor的函数，但不共享内存，转换较慢
            points = torch.Tensor(points)

            #装cuda里
            points, target = points.float().cuda(), target.long().cuda()
            #1，2维度转置 B*N*C 变 B*C*N
            points = points.transpose(2, 1)
            
            #预测
            #print("Type of inputdata:",points.size())
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) #修改形状 某一维度*Num_Classes 大小

            #目标值转成某一维度*1的大小，取全部 放CPU里，转numpy
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            #目标值转成某一维度*1的大小，取全部
            target = target.view(-1, 1)[:, 0]

            loss = criterion(seg_pred, target, trans_feat)#, weights)
            
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        #log_string('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training mean loss: %f' % (loss_sum / num_batches))
        #log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))
        
        if epoch % 5 == 0:
            print("Save model")
            #logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            #log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            #log_string('Saving model....')

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
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat)#, weights)
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
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

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
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
