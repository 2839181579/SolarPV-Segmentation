import os
import datetime
import glob

# -----------------------------------------------------------------------------------
# [配置] 权重下载路径
# -----------------------------------------------------------------------------------
custom_weight_path = r"E:\BaiduNetdiskDownload\UNet_Demo\model_data"
if not os.path.exists(custom_weight_path):
    os.makedirs(custom_weight_path)
os.environ['TORCH_HOME'] = custom_weight_path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入 SMP
import segmentation_models_pytorch as smp

from nets.unet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ---------------------------------#
    #   基础配置
    # ---------------------------------#
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = True
    num_classes = 2

    # -----------------------------------------------------#
    #   [修改1] 更名为 mit_b5
    #   MiT-B5 是 SegFormer 系列中最强的主干之一
    # -----------------------------------------------------#
    backbone = "mit_b5"
    pretrained = False
    model_path = ""  # 设为空，从头加载 ImageNet 预训练权重

    # -----------------------------------------------------#
    #   input_shape
    #   显存不够时，可以尝试从 512 改为 480 或 448
    # -----------------------------------------------------#
    input_shape = [512, 512]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练参数设置
    # ----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 80
    Freeze_batch_size = 16  # 4090 显存大，冻结阶段跑 16 没问题

    UnFreeze_Epoch = 200

    # -----------------------------------------------------#
    #   [修改2] 解冻后 BatchSize 设为 4
    #   mit_b5 参数量大，解冻后梯度图占用极高。
    #   如果 4090 跑 4 还有余力，可以尝试改为 6 或 8
    # -----------------------------------------------------#
    Unfreeze_batch_size = 4

    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   [修改3] 学习率微调
    #   为了避免解冻后 Loss 暴涨不回落，这里设置较小的 Min_lr
    # ------------------------------------------------------------------#
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 0.01

    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # 数据集路径
    VOCdevkit_path = 'VOCdevkit_Competition'

    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    # ------------------------------------------------------#
    #   设备设置
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # ----------------------------------------------------#
    #   [修改4] 构建 MiT-B5-UNet 模型
    # ----------------------------------------------------#
    if local_rank == 0:
        print(f"正在构建 MiT-B5-UNet 模型 (Backbone: {backbone})...")

    model = smp.Unet(
        encoder_name="mit_b5",  # 使用 B5
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).train()

    # 加载断点权重逻辑 (如有)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 读取数据集
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    # ---------------------------------------#
    #   训练循环
    # ---------------------------------------#
    if True:
        UnFreeze_flag = False

        # 冻结阶段
        if Freeze_Train:
            for param in model.encoder.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        optimizer = optim.AdamW(model.parameters(), lr=Init_lr, weight_decay=weight_decay)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

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
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 解冻逻辑
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                for param in model.encoder.parameters():
                    param.requires_grad = True

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

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

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss,
                          focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()

            # ----------------------------------------------------#
            #   [新增] 训练结束后，自动分析 Log 寻找最佳结果
            # ----------------------------------------------------#
            print("\n" + "=" * 50)
            print("训练结束，正在分析最优模型指标...")
            print("=" * 50)

            # 这里的路径是 loss_history 创建的路径
            target_log_dir = log_dir

            # 尝试寻找 mIoU 文件
            miou_file = os.path.join(target_log_dir, "epoch_miou.txt")
            mpa_file = os.path.join(target_log_dir, "epoch_mpa.txt")  # 如果你的utils里没写这个，可能找不到
            acc_file = os.path.join(target_log_dir, "epoch_accuracy.txt")

            best_miou = 0
            best_epoch = -1

            if os.path.exists(miou_file):
                with open(miou_file, 'r') as f:
                    # 读取所有行，跳过第一行(如果有标题的话，看你的txt格式第一行好像是0)
                    lines = f.readlines()
                    # 你的txt格式是每行一个数字，第一行是0或者初始值
                    # 我们遍历所有行找到最大值
                    for idx, line in enumerate(lines):
                        try:
                            val = float(line.strip())
                            if val > best_miou:
                                best_miou = val
                                best_epoch = idx + 1  # 假设第一行对应 epoch 1
                        except:
                            continue

                print(f"【最佳表现】 Found Best mIoU: {best_miou:.2f}% at Epoch {best_epoch}")

                # 尝试读取同 Epoch 的 mPA 和 Accuracy (如果文件存在)
                if os.path.exists(mpa_file):
                    with open(mpa_file, 'r') as f:
                        mpa_lines = f.readlines()
                        if best_epoch <= len(mpa_lines):
                            print(f"           Corresponding mPA : {float(mpa_lines[best_epoch - 1].strip()):.2f}%")

                if os.path.exists(acc_file):
                    with open(acc_file, 'r') as f:
                        acc_lines = f.readlines()
                        if best_epoch <= len(acc_lines):
                            print(f"           Corresponding Acc : {float(acc_lines[best_epoch - 1].strip()):.2f}%")

                print("\n提示：请去 logs 文件夹下查看具体的 epoch_xxx.txt 文件以获取完整数据。")
                print(f"日志保存路径: {target_log_dir}")

            else:
                print(f"未找到 mIoU 日志文件: {miou_file}")
                print("请检查 utils/callbacks.py 是否正确写入了 epoch_miou.txt")