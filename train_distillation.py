import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader


# teacher network
from nets.UNet.swinTS_Att_Unet import swinTS_Att_Unet

# student network
from nets.FastSegFormer.fast_segformer import FastSegFormer

from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch_distillation


if __name__ == "__main__":
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = False
    num_classes = 3 + 1
    """
    teacher backbone = "swin_T_224" or "swin_S_224" or "poolformer_s24" or "poolformer_s36"
    student backbone = "poolformer_s12" or "efficientformerV2_s0"
    """
    t_backbone = "swin_T_224"
    s_backbone = "poolformer_s12"
    pretrained = False
    t_model_path = "model_data/teacher_Swin_T_Att_Unet_input_512.pth"
    s_model_path = "" # if s_model_path = "": from scratch else fine-tuning
    """
    if Multi_resolution = True, the input_shape should be teacher network's input shape and the student network's shape is default [224, 224]
    if Multi_resolution = False, the input_shape is the shape of the teacher network and the student network
    """
    input_shape = [512, 512]
    Multi_resolution = True
    # input_shape = [224, 224]
    # Multi_resolution = False

    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 6
    UnFreeze_Epoch = 1000
    Unfreeze_batch_size = 6
    Freeze_Train = False

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0

    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = 'Orange_Navel_1.5k'

    NFD_loss = True
    KL_pixel_loss = True
    dice_loss = False
    focal_loss = False

    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 1
    ngpus_per_node = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(t_backbone)
                download_weights(s_backbone)
            dist.barrier()
        else:
            download_weights(t_backbone)
            download_weights(s_backbone)


    """
    network
    """
    # student model
    s_model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=s_backbone, Pyramid="multiscale", fork_feat=True, cnn_branch=True)


    # teacher model
    # t_model = FastSegFormer(num_classes=num_classes, pretrained=pretrained, backbone=t_backbone, Pyramid="multiscale", fork_feat=True, cnn_branch=True)
    t_model = swinTS_Att_Unet(num_classes=num_classes, pretrained=pretrained, backbone=t_backbone, fork_feat=True)

    if not pretrained:
        weights_init(s_model)
    if t_model_path != '':
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load teacher network weights {}.'.format(t_model_path))

        t_model_dict = t_model.state_dict()
        t_load_key, t_no_load_key, t_temp_dict = [], [], {}
        t_pretrained_dict = torch.load(t_model_path, map_location=device)

        for k, v in t_pretrained_dict.items():
            if k in t_model_dict.keys() and np.shape(t_model_dict[k]) == np.shape(v):
                t_temp_dict[k] = v
                t_load_key.append(k)
            else:
                t_no_load_key.append(k)
        t_model_dict.update(t_temp_dict)
        t_model.load_state_dict(t_model_dict)
        if local_rank == 0:
            print("\nTeacher network successful Load Key:", str(t_load_key)[:500], "……\nSuccessful Load Key Num:", len(t_load_key))
            print("\nTeacher network fail To Load Key:", str(t_no_load_key)[:500], "……\nFail To Load Key num:", len(t_no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    if s_model_path != '':
        if local_rank == 0:
            print('Load student backbone network weights {}.'.format(s_model_path))

        s_model_dict = s_model.state_dict()
        s_load_key, s_no_load_key, s_temp_dict = [], [], {}

        if s_backbone == "efficientformerV2_s0":
            s_pretrained_dict = torch.load(s_model_path, map_location=device)['model']
        else:
            s_pretrained_dict = torch.load(s_model_path, map_location=device)


        s_backbone_stat_dict = {}  # 修改的权重字典
        for i in s_pretrained_dict.keys():
            s_backbone_stat_dict["common_backbone." + i] = s_pretrained_dict[i]
        s_pretrained_dict.update(s_backbone_stat_dict)  # 更新权重


        for k, v in s_pretrained_dict.items():
            if k in s_model_dict.keys() and np.shape(s_model_dict[k]) == np.shape(v):
                s_temp_dict[k] = v
                s_load_key.append(k)
            else:
                s_no_load_key.append(k)
        s_model_dict.update(s_temp_dict)
        s_model.load_state_dict(s_model_dict)
        if local_rank == 0:
            print("\nStudent backbone successful Load Key:", str(s_load_key)[:500], "……\nSuccessful Load Key Num:", len(s_load_key))
            print("\nStudent backbone fail To Load Key:", str(s_no_load_key)[:500], "……\nFail To Load Key num:", len(s_no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "Stu_loss_" + str(time_str))
        loss_history = LossHistory(log_dir, s_model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    t_model_eval = t_model.eval()
    s_model_train = s_model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        t_model_eval = torch.nn.SyncBatchNorm.convert_sync_batchnorm(t_model_eval)
        s_model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(s_model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            s_model_train = s_model_train.cuda(local_rank)
            s_model_train = torch.nn.parallel.DistributedDataParallel(s_model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            s_model_train = torch.nn.DataParallel(s_model)
            t_model_eval = torch.nn.DataParallel(t_model)
            cudnn.benchmark = True
            s_model_train = s_model_train.cuda()
            t_model_eval = t_model_eval.cuda()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        print("Student network config:\n")
        show_config(
            num_classes=num_classes, backbone=s_backbone, model_path=s_model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            s_model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(s_model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(s_model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("dataset is small, please expand the dataset!")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback = EvalCallback(Multi_resolution, s_model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period, distillation=True)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                s_model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("dataset is small, please expand the dataset!")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch_distillation(Multi_resolution, s_model_train, s_model, t_model_eval, t_model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, NFD_loss, KL_pixel_loss, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
