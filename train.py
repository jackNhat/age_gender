import os
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import get_config
from backbone.model_irse import IR_50, IR_152
from backbone.model_resnet import ResNet_50
from backbone.model_mobilefacenet import MobileFaceNet
from head.metrics import SFaceLoss, Am_softmax, ArcFace, Softmax, CosFace, SphereFace
from dataset import make_datasets, make_frame

from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_time, AverageMeter, train_accuracy


def xavier_normal_(tensor, gain=1., mode='avg'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'avg':
        fan = fan_in + fan_out
    elif mode == 'in':
        fan = fan_in
    elif mode == 'out':
        fan = fan_out
    else:
        raise Exception('wrong mode')
    std = gain * math.sqrt(2.0 / float(fan))

    return nn.init._no_grad_normal_(tensor, 0., std)


def weight_init(m):
    #print(m)
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.fill_(1)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()
        if hasattr(m, 'running_mean') and m.running_mean is not None:
            m.running_mean.data.zero_()
        if hasattr(m, 'running_var') and m.running_var is not None:
            m.running_var.data.fill_(1)
    elif isinstance(m, nn.PReLU):
        m.weight.data.fill_(1)
    else:
        if hasattr(m, 'weight') and m.weight is not None:
            xavier_normal_(m.weight.data, gain=2, mode='out')
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--workers_id', help="gpu ids or cpu", default='1', type=str)
    parser.add_argument('--epochs', help="training epochs", default=128, type=int)
    parser.add_argument('--stages', help="training stages", default='35,65,95', type=str)
    parser.add_argument('--lr',help='learning rate',default=1e-1, type=float)
    parser.add_argument('--batch_size', help="batch_size", default=64, type=int)
    # parser.add_argument('--data_mode', help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='ms1m', type=str)
    parser.add_argument('--net', help="which network, ['IR_50', 'Res_50', 'MobileFaceNet']",default='Res_50', type=str)
    parser.add_argument('--head', help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax','SFaceLoss']", default='ArcFace', type=str)
    parser.add_argument('--target', help="verification targets", default='agedb_30', type=str)
    parser.add_argument('--resume_backbone', help="resume backbone model", default='', type=str)
    parser.add_argument('--resume_head', help="resume head model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='test_dir', type=str)
    parser.add_argument('--param_s', default=64.0, type=float)
    parser.add_argument('--param_k', default=80.0, type=float)
    parser.add_argument('--param_a', default=0.8, type=float)
    parser.add_argument('--param_b', default=1.23, type=float)
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    CSV_FILE = cfg['CSV_FILE']
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train data are stored
    NUM_CLASS = cfg['NUM_CLASS']
    EVAL_PATH = cfg['EVAL_PATH'] # the parent root where your val data are stored
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['IR_50', 'IR_101']
    HEAD_NAME = cfg['HEAD_NAME']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)

    writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        # NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    # assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]
    
    frames = make_frame(CSV_FILE, DATA_ROOT)
    train_data = make_datasets(frames, input_size = INPUT_SIZE, give_dataloader=False, col_used='ageAndgender')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'IR_50': IR_50(INPUT_SIZE),
                    'Res_50': ResNet_50(INPUT_SIZE),
                     'IR_152': IR_152(INPUT_SIZE),
                     'MobileFaceNet': MobileFaceNet(EMBEDDING_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    # HEAD = SFaceLoss(in_features = EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID,
    #                  s = args.param_s, k = args.param_k, a = args.param_a, b = args.param_b)
    HEAD_DICT = {'Softmax': Softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SFaceLoss': SFaceLoss(in_features = EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID, s = args.param_s, 
                 k = args.param_k, a = args.param_a, b = args.param_b)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)
    print("=" * 60)
    print(HEAD)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    elif BACKBONE_NAME.find("MobileFace") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_mobilefacenet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_mobilefacenet_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    BACKBONE.apply(weight_init)
    HEAD.apply(weight_init)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT,HEAD_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        elif os.path.isfile(HEAD_RESUME_ROOT):    
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = 200 # frequency to display training loss & acc
    batch = 0  # batch index

    intra_losses = AverageMeter()
    inter_losses = AverageMeter()
    Wyi_mean = AverageMeter()
    Wj_mean = AverageMeter()
    top1 = AverageMeter()

    BACKBONE.train()  # set to training mode
    HEAD.train()
    
    highest_acc = 0.0

    for epoch in range(NUM_EPOCH):

        if epoch in STAGES:
            schedule_lr(OPTIMIZER)

        last_time = time.time()

        for inputs, labels in iter(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)

            outputs, loss, intra_loss, inter_loss, WyiX, WjX = HEAD(features, labels)

            prec1 = train_accuracy(outputs.data, labels, topk=(1,))
            #embed()
            intra_losses.update(intra_loss.data.item(), inputs.size(0))
            inter_losses.update(inter_loss.data.item(), inputs.size(0))
            Wyi_mean.update(WyiX.data.item(), inputs.size(0))
            Wj_mean.update(WjX.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                intra_epoch_loss = intra_losses.avg
                inter_epoch_loss = inter_losses.avg
                Wyi_record = Wyi_mean.avg
                Wj_record = Wj_mean.avg
                epoch_acc = top1.avg
                writer.add_scalar("intra_Loss", intra_epoch_loss, batch + 1)
                writer.add_scalar("inter_Loss", inter_epoch_loss, batch + 1)
                writer.add_scalar("Wyi", Wyi_record, batch + 1)
                writer.add_scalar("Wj", Wj_record, batch + 1)
                writer.add_scalar("Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'intra_Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'inter_Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Wyi {Wyi.val:.4f} ({Wyi.avg:.4f})\t'
                      'Wj {Wj.val:.4f} ({Wj.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss1 = intra_losses, loss2 = inter_losses, Wyi=Wyi_mean, Wj=Wj_mean, top1=top1))

                # save checkpoints per epoch
                if top1.avg > highest_acc:
                    highest_acc = top1.avg
                    print('saved model with highest acc: ', highest_acc)
                    torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH,
                    "Best_Backbone_{}_checkpoint.pth".format(BACKBONE_NAME)))
                    torch.save(HEAD.state_dict(), os.path.join(WORK_PATH,
                    "Best_Head_{}_checkpoint.pth".format(HEAD_NAME)))
            batch += 1  # batch index
