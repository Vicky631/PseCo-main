#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------
# @Description: 基于SAM-ViT-H预训练分割模型+PointDecoder点解码模型+CLIP-ViT-B/32跨模态模型，分布式推理实现FSC147稠密目标计数数据集的目标预测与特征提取；加载预处理的图像特征与标注数据，通过点解码器生成目标候选点，结合SAM完成目标框预测，再利用CLIP提取候选框图像特征，最终合并分布式推理结果并统一保存，全程无训练，纯GPU推理加速，适配多卡分布式运行。
# @Input:
# 1. 预处理特征文件: /home/zzhuang/PseCo/data/fsc147/sam/all_data_vith.pth，包含图像SAM特征、标注信息、图像宽高、mask掩码、数据划分(train/val/test)等；
# 2. 模型权重文件: /home/zzhuang/PseCo/data/fsc147/checkpoints/point_decoder_vith.pth (PointDecoder模型权重)；
# 3. 预训练模型: SAM-ViT-H分割模型、CLIP-ViT-B/32跨模态匹配模型；
# 4. 原始图像文件: /home/zzhuang/PseCo/data/fsc147/images_384_VarV2/ 下的FSC147数据集JPG图像；
# 5. 代码依赖文件: 自定义PointDecoder模型、SAM/CLIP相关算子、图像处理及分布式相关依赖库。
# @Output:
# 1. /home/zzhuang/PseCo/data/fsc147/sam/all_predictions_vith_{rank}.pth: 各分布式进程单独保存的推理结果文件，字典格式，键为图像名，值为单张图像的预测数据；
# 2. /home/zzhuang/PseCo/data/fsc147/sam/all_predictions_vith.pth: 合并所有分布式进程后的最终完整推理结果文件，字典格式，键为图像文件名，值包含：pred_boxes(SAM预测的目标边界框张量)、pred_ious(预测框置信度分数张量)、pred_points_score(候选点预测得分张量)、pred_points(点解码器生成的目标候选点张量)、clip_regions(字典，含clip_embeddings候选框CLIP图像特征、boxes采样后的预测框)。
# ----------------------------------------------------------------
import os
import sys

sys.path.insert(0, '/home/zzhuang/PseCo')
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import tqdm
import albumentations as A
import torch.nn as nn
import torchvision
import torchvision.ops as vision_ops
from ops.foundation_models.segment_anything.utils.amg import batched_mask_to_box
from ops.ops import _nms, plot_results, convert_to_cuda
import json

torch.autograd.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')
from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, \
    build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

# GPU分布式训练设置
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
opt = parser.parse_args()
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')

torch.cuda.set_device(dist.get_rank())


def chunks(arr, m):
    import math
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


# 模型加载
from models import PointDecoder

project_root = '/home/zzhuang/PseCo'

sam = build_sam_vit_h().cuda().eval()
point_decoder = PointDecoder(sam).cuda().eval()
state_dict = torch.load(f'{project_root}/data/fsc147/checkpoints/point_decoder_vith.pth',
                        map_location='cpu')
point_decoder.load_state_dict(state_dict)
# 数据加载
all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith.pth', map_location='cpu')
# 结果保存路径
base_path = f'{project_root}/data/fsc147/sam/all_predictions_vith'

# sam = build_sam_vit_l().cuda().eval()
# point_decoder = PointDecoder(sam).cuda().eval()
# state_dict = torch.load(f'{project_root}/data/fsc147/checkpoints/point_decoder_vitl.pth',
#                         map_location='cpu')['point_decoder']
# point_decoder.load_state_dict(state_dict)
# all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vitl.pth', map_location='cpu')
# base_path = f'{project_root}/data/fsc147/sam/all_predictions_vitl'

# 为 PointDecoder 准备输入
# 1. 为每张图像生成256x256的掩码（Mask）
for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    target['image_id'] = fname
    transform = A.Compose([
        A.LongestMaxSize(256),
        A.PadIfNeeded(256, 256, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    mask = Image.fromarray(
        transform(image=np.ones((target['height'], target['width'])).astype(np.uint8) * 255)['image'])
    mask = np.array(mask) > 128
    target['mask'] = torch.from_numpy(mask).reshape(1, 1, 256, 256).bool().float()
# 2. 划分数据集（train/val/test/all）
all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    # if all_data[fname]['split'] == 'train':
    #     if (all_data[fname]['annotations']['points'].size(0) + all_data[fname]['segment_anything']['points'].size(
    #             0)) != 0:
    #         all_image_list[all_data[fname]['split']].append(fname)
    # else:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)
# 3. 为当前进程分配图像列表（数据分片）
all_file_names = chunks(all_image_list['all'], dist.get_world_size())[dist.get_rank()]
# 4. 初始化推理结果字典（如果已有部分结果，直接加载避免重复计算）
save_path = f'{base_path}_{dist.get_rank()}.pth'

if os.path.exists(save_path):
    predictions = torch.load(save_path, map_location='cpu')
else:
    predictions = {}

# ===================== 核心推理流程（PointDecoder+SAM 生成目标点和框） =====================

print(dist.get_rank(), len(all_file_names))
for n_iter, fname in enumerate(tqdm.tqdm(all_file_names)):
    if fname in predictions:
        continue
    # 1. 加载当前图像的SAM特征 → 转GPU
    features = all_data[fname]['features'].cuda()
    with torch.no_grad():
        # point_decoder.max_points = 256
        # point_decoder.point_threshold = 0.05
        # point_decoder.nms_kernel_size = 5
        # 配置PointDecoder参数：最大候选点数量、置信度阈值、NMS核大小
        point_decoder.max_points = 2000
        point_decoder.point_threshold = 0.01
        point_decoder.nms_kernel_size = 3
        # 2. PointDecoder推理：输入SAM特征+Mask → 输出目标点热力图
        outputs_heatmaps = point_decoder(features, masks=all_data[fname]['mask'].cuda())

    # 3. 解析PointDecoder输出：候选点+置信度分数
    pred_points = outputs_heatmaps['pred_points'].squeeze()  # [N,2] 目标点坐标
    pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()  # [N] 点置信度

    # 4. SAM基于候选点生成目标框
    all_pred_boxes = []
    all_pred_scores = []
    # 分批处理候选点（避免GPU显存溢出）
    for indices in torch.arange(len(pred_points)).split(256):
        with torch.no_grad():
            # 4.1 SAM点提示推理：输入特征+候选点 → 输出目标框和IoU
            outputs_points = sam.forward_sam_with_embeddings(features, points=pred_points[indices].reshape(-1, 2))
            pred_boxes = outputs_points['pred_boxes']  # [B,4] 目标框（xyxy）
            pred_logits = outputs_points['pred_ious']  # [B,1] 框置信度

            # 4.2 SAM锚框提示推理（额外提升框精度）
            for anchor_size in [8, ]:
                # 生成锚框：以候选点为中心，8x8大小
                anchor = torch.Tensor([[-anchor_size, -anchor_size, anchor_size, anchor_size]]).cuda()
                anchor_boxes = pred_points[indices].reshape(-1, 2).repeat(1, 2) + anchor
                anchor_boxes = anchor_boxes.clamp(0., 1024.)  # 限制在图像范围内
                # SAM框提示推理
                outputs_boxes = sam.forward_sam_with_embeddings(features, points=pred_points[indices].reshape(-1, 2),
                                                                boxes=anchor_boxes)
                # 合并点提示和框提示的结果
                pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
                pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_logits)

    # 5. 合并所有批次的结果 → 转CPU保存
    pred_boxes = torch.cat(all_pred_boxes).cpu()
    pred_scores = torch.cat(all_pred_scores).cpu()

    # 6. 保存当前图像的推理结果
    predictions[fname] = {
        'pred_boxes': pred_boxes,
        'pred_ious': pred_scores,
        'pred_points_score': pred_points_score,
        'pred_points': pred_points,
    }
    # 每100张图像保存一次，防止意外中断
    if n_iter % 100 == 0:
        torch.save(predictions, save_path)
# torch.save(predictions, save_path)

# ===================== CLIP 特征提取 + 分布式结果合并（最终输出） =====================

from ops.foundation_models import clip
import numpy as np

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
import torchvision

normalize = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711))


def read_image(fname):
    img = Image.open(f'{project_root}/data/fsc147/images_384_VarV2/{fname}')
    transform = A.Compose([
        A.LongestMaxSize(1024),
        A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    img = Image.fromarray(transform(image=np.array(img))['image'])
    return img


def extract_clip_features(fname, bboxes):
    image = read_image(fname)
    examples = []
    for box in bboxes:
        example = image.crop(box.long().tolist())
        example = example.resize((224, 224))
        example = normalize(to_tensor(example)).unsqueeze(0)
        examples.append(example)
    examples = torch.cat(examples)
    e = []
    with torch.no_grad():
        for indices in torch.arange(len(examples)).split(256):
            e.append(model.encode_image(examples[indices].cuda()).float())
    e = torch.cat(e, dim=0)
    e = F.normalize(e, dim=1).cpu()
    return e

# 入口：为每个图像的候选框提取CLIP特征
for n_iter, fname in enumerate(tqdm.tqdm(all_file_names)):
    if 'clip_regions' in predictions[fname]:
        continue
    print(fname)
    pred_boxes = predictions[fname]['pred_boxes'].cuda()
    pred_points_score = predictions[fname]['pred_points_score'].cuda()
    # 按置信度采样256个候选框（避免特征过多）
    rand_indices = torch.multinomial(pred_points_score, min(len(pred_boxes), 256), replacement=False)
    boxes = pred_boxes[rand_indices]
    # 提取CLIP特征并保存
    predictions[fname]['clip_regions'] = {
        'clip_embeddings': extract_clip_features(fname, boxes.reshape(-1, 4)).view(-1, boxes.size(1), 512),
        'boxes': boxes,
    }
    if n_iter % 100 == 0:
        torch.save(predictions, save_path)
torch.save(predictions, save_path)

# ===================== 合并所有GPU进程的结果 =====================

predictions = {}
for i in range(dist.get_world_size()):
    data = torch.load(f'{base_path}_{i}.pth', map_location='cpu')
    for fname in data:
        predictions[fname] = data[fname]
torch.save(predictions, base_path + '.pth')

# break
