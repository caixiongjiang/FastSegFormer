import os
import datetime
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.FastSCNN.fast_scnn import FastSCNN                 # pretrained = False model_path is None (no pretrained)
from nets.FANet.fa_net import FANet                          # pretrained = False model_path is None
from nets.FastSegFormer.fast_segformer import FastSegFormer  # pretrained = False model_path = ""
from nets.SwiftNet.swiftnet import SwiftNet                  # pretrained = False model_path is None
from nets.BiseNet.bisenet import BiSeNet                     # pretrained = False model_path is None
from nets.ENet.enet import ENet                              # pretrained = False model_path is None (no pretrained)
from nets.PIDNet.pidnet import get_pred_model                # pretrained = False model_path = ""
from nets.UNet.swinTS_Att_Unet import swinTS_Att_Unet        # pretrained = False model_path = ""


from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch



if __name__ == "__main__":
    Cuda = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    num_classes = 3 + 1

    """
    backbone:
    resnet18, resnet34, resnet50, resnet101, resnet152
    swin_T_224, swin_S_224
    poolformer_s12, poolformer_s24, poolformer_s36
    efficientformerV2_s0
    if backbone == None: backbone = ""
    """

    backbone    = "poolformer_s12"
    pretrained  = False
    model_path  = "model_data/poolformer_s12.pth"
    input_shape = [224, 224]

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    UnFreeze_Epoch      = 1000
    Unfreeze_batch_size = 32
    Freeze_Train        = False

    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 5

    VOCdevkit_path  = 'Orange_Navel_1.5k'
    dice_loss       = False
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 1
    ngpus_per_node  = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)


    """
    fast network
    """
    # Other models
    # model = FANet(nclass=num_classes, backbone=backbone)
    # model = FastSCNN(num_classes=num_classes)
    # model = SwiftNet(num_classes=num_classes)
    # model = BiSeNet(nclass=num_classes, backbone=backbone, spatial_path=True)
    # model = ENet(4)
    # model = get_pred_model('pidnet_s', num_classes=num_classes)
    # model = get_pred_model('pidnet_m', num_classes=num_classes)
    # model = get_pred_model('pidnet_l', num_classes=num_classes)


    # student model
    # model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="pooling")
    # model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="multiscale")
    model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="multiscale", cnn_branch=True)

    # teacher model
    # model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="multiscale")
    # model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="multiscale")
    # model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=backbone, Pyramid="multiscale", cnn_branch=True)
    # model = swinTS_Att_Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)


    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict      = model.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        if backbone == "resnet50" or backbone == "none":
            pretrained_dict = torch.load(model_path, map_location=device)

        elif backbone == 'PIDNet_S' or backbone == 'PIDNet_M' or backbone == 'PIDNet_L':
            pretrained_dict = torch.load(model_path, map_location=device)['state_dict']

        elif backbone == "efficientformerV2_s0" or backbone == "swin_T_224" or backbone == "swin_S_224":
            pretrained_dict = torch.load(model_path, map_location=device)['model']
            backbone_stat_dict = {}
            for i in pretrained_dict.keys():
                backbone_stat_dict["common_backbone." + i] = pretrained_dict[i]
            pretrained_dict.update(backbone_stat_dict)

        else:
            pretrained_dict = torch.load(model_path, map_location=device)
            backbone_stat_dict = {}
            for i in pretrained_dict.keys():
                backbone_stat_dict["common_backbone." + i] = pretrained_dict[i]
            pretrained_dict.update(backbone_stat_dict)

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape,
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type,
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("dataset is small, please expand the dataset!")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("dataset is small, please expand the dataset!")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
