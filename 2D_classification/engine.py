# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time
import os
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from utils import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score


from sklearn.metrics import roc_auc_score

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGDAttack(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit, max_iters, random_init):
    model.eval()

    delta = torch.zeros_like(x).cuda()
    if random_init:
        for iiiii in range(len(attack_epsilon)):
            delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())
    
    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    max_iters = int(max_iters)
    adv_imgs.requires_grad = True 

    with torch.enable_grad():
        for _iter in range(max_iters):
            
            outputs = model(adv_imgs)

            loss = loss_fn(outputs, y)

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            adv_imgs.data += attack_alpha * torch.sign(grads.data) 
            
            adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)

            adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)

    return adv_imgs.detach()

def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,16,16)
    patches = patch_transform(patches)
 
    patches = patches.reshape(bs, -1, channle_size,16,16).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*16*16, -1)
    output_images = F.fold(patches, (H,W), 16, stride=16)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images


def train_one_epoch(args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_parameters,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, use_wandb=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", use_wandb=use_wandb)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # if i % 100 == 0:
        #     model.module.update_L(lam=0.1)
        # i = i + 1

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast():
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                if args.recon_loss > 1e-4:
                    loss = loss + args.recon_loss * model.module.recon_loss()
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # normalize the FFT kernel
        # model.module.normalize_parameters()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

@torch.no_grad()
def evaluate(data_loader, model, device, mask=None, adv=None, prefix=""):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_predictions, all_labels = [], []
    for images, target in data_loader:
        all_labels.extend(target.cpu().numpy())
    all_labels = np.array(all_labels)
    unique_numbers = set(all_labels)
    unique_numbers_length = len(unique_numbers)

    for images, target in data_loader: #metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (0.5 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                output = model(images)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            probabilities = torch.softmax(output[:,:unique_numbers_length], dim=1, dtype=torch.double)
            all_predictions.extend(probabilities.cpu().numpy())
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    all_predictions = np.array(all_predictions)
    if unique_numbers_length == 2:
        auc = roc_auc_score(all_labels, all_predictions[:, 1]) * 100
    else:
        auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr') * 100

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* ' + prefix + ': ACC@1 {top1.global_avg:.3f} ACC@5 {top5.global_avg:.3f} AUC {auc:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, auc=auc, losses=metric_logger.loss))
    res_map = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    res_map["auc"] = auc
    return res_map


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def evaluate_cam(data_loader, model, device, model_name, dataset_path):

    root_path = "./datasets/camSample/" + dataset_path
    output_path = f"./datasets/camSample/{dataset_path}/output/{model_name}"
    # Check if the folder does not exist
    if not os.path.exists(output_path):
        # Create the folder
        os.makedirs(output_path)
        print("Folder created:", output_path)
    else:
        print("Folder already exists:", output_path)

    for index, (images, target) in enumerate(data_loader):
        images = images.to(device)

        input_tensor = images
        cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)
        # cam = GradCAM(model=model, target_layers=[model.llama_dim_mapper1], reshape_transform=reshape_transform)

        target_category = None  # can be a class or none
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
        grayscale_cam = grayscale_cam[0, :]

        # add grad-cam to original image
        rgb_img = cv2.imread(root_path + "/test/0/" + str(index) + ".jpg", 1)[:, :, ::-1]
        rgb_img_normalized = (rgb_img.astype(np.float32) / 255.0).clip(0, 1)
        rgb_img_normalized_resized = cv2.resize(rgb_img_normalized, (224, 224))
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))

        # Visualization function should be called here to overlay heatmap on the original image
        visualization = show_cam_on_image(rgb_img_normalized_resized, grayscale_cam)

        # Save visualization image
        cv2.imwrite(f'{output_path}/cam_{str(index)}.jpg', visualization)
        cv2.imwrite(f'{output_path}/origin_{str(index)}.jpg', rgb_img_resized)