#!/usr/bin/env python
# coding: utf-8

# ----------------------------------------------------------------
# @File Description: 基于SAM(Segment Anything Model-ViT-H)与CLIP(ViT-B/32)实现FSC147稠密目标计数数据集的全流程预处理，完成数据集标注校验、图像预处理、SAM图像特征提取、目标候选框生成、高质量伪标签构建、CLIP跨模态图像/文本特征提取，为后续稠密目标计数模型训练提供全量预处理数据，全程为推理流程，基于GPU加速，无模型训练步骤。
# @Input:
# 1. 图像数据: /home/zy/wjj/PseCo-main/data/fsc147/images_384_VarV2/ 下的FSC147数据集JPG格式原始图像；
# 2. 标注文件: /home/zy/wjj/PseCo-main/data/fsc147/annotation_FSC147_384_with_gt.json，包含图像宽高、类别名、目标标注点、示例框、真实计数的json标注；
# 3. 预训练模型: SAM-ViT-H分割模型、CLIP-ViT-B/32跨模态匹配模型。
# @Output:
# 1. /home/zy/wjj/PseCo-main/data/fsc147/sam/all_data_vith.pth: 字典格式，键为图像名，值包含图像缩放后的标注点、示例框、宽高、类别名、SAM图像特征、CLIP示例框图像特征；
# 2. /home/zy/wjj/PseCo-main/data/fsc147/sam/segment_anything_data_vith.pth: 字典格式，键为图像名，值包含SAM网格点推理后过滤的目标预测框、掩码轮廓中心点、预测IoU置信度；
# 3. /home/zy/wjj/PseCo-main/data/fsc147/sam/pseudo_boxes_data_vith.pth: 字典格式，键为图像名，值为SAM真实标注点推理生成的高质量目标伪标签边界框；
# 4. /home/zy/wjj/PseCo-main/data/fsc147/clip_text_prompt.pth: 字典格式，键为数据集类别名，值为对应类别的归一化CLIP文本特征向量(512维)。
# ----------------------------------------------------------------

import os
import sys

sys.path.insert(0, '/home/zy/wjj/PseCo-main')
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
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

plt.rcParams["figure.dpi"] = 300
import json

torch.cuda.set_device(0)
torch.autograd.set_grad_enabled(False)
# get_ipython().system('gpustat')

# ===================== 可视化校验数据集标注准确行性 =====================

import sys

sys.path.append('/home/zy/wjj/PseCo-main')

import json
import tqdm
from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, \
    build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

# 初始化SAM模型，使用vit_h版本并移动到GPU上进行推理
sam = build_sam_vit_h().cuda().eval()

# 定义项目根目录路径
project_root = '/home/zy/wjj/PseCo-main/'


def read_image(fname):
    """
    读取并预处理图像

    该函数从指定路径读取图像，然后应用变换操作（调整最大尺寸为1024，
    并填充到1024x1024尺寸）以适应模型输入要求

    Args:
        fname (str): 图像文件名

    Returns:
        PIL.Image: 预处理后的图像对象
    """
    img = Image.open(f'{project_root}/data/fsc147/images_384_VarV2/{fname}')
    transform = A.Compose([
        A.LongestMaxSize(1024),
        A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    img = Image.fromarray(transform(image=np.array(img))['image'])
    return img


# 加载FSC147数据集的标注信息
all_data = json.load(open(f'{project_root}/data/fsc147/annotation_FSC147_384_with_gt.json'))
# 遍历所有数据，对标注信息进行尺度变换以适应1024的最大尺寸
for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    scale = 1024 / max(target['width'], target['height'])
    target['annotations'] = {
        'points': torch.Tensor([target['annotations'][l]['points'] for l in target['annotations']]).float() * scale}
    target['box_examples_coordinates'] = torch.Tensor(target['box_examples_coordinates']).float() * scale

# 加载并显示示例图像及其标注信息
fname = '2.jpg'
plot_results(read_image(fname),
             points=all_data[fname]['annotations']['points'],
             bboxes=all_data[fname]['box_examples_coordinates']
             )
# ===================== SAM图像编码器生成图像特征向量 =====================


import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
for fname in tqdm.tqdm(all_data):
    image = read_image(fname)
    with torch.no_grad():
        new_image = transform(image).unsqueeze(0).cuda()
        features = sam.image_encoder(new_image)
    all_data[fname]['features'] = features.cpu()

os.makedirs(f'{project_root}/data/fsc147/sam/', exist_ok=True)

# torch.save(all_data, f'{project_root}/data/fsc147/sam/all_data_vitl.pth')
torch.save(all_data, f'{project_root}/data/fsc147/sam/all_data_vith.pth')
all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith.pth', map_location='cpu')

# ===================== 基于网格点预测mask =====================


from ops.foundation_models.segment_anything.utils.amg import build_point_grid, calculate_stability_score

# 1. 设置网格每边的点数，平衡精度和计算效率
# points_per_side = 64
points_per_side = 32
# points_per_side = 16
grid_points = build_point_grid(points_per_side)

# 2.初始化结果变量
segment_anything_data = {}

# 3. 遍历所有数据，对每个图像生成网格点
for fname in tqdm.tqdm(all_data):
    # features = torch.load('/home/zzhuang/DATASET2/COCO/sam_vith/Org_features/' + fname, map_location='cpu').cuda()
    features = all_data[fname]['features'].cuda()
    h, w = all_data[fname]['height'], all_data[fname]['width']
    grid_points_ = grid_points.copy()
    grid_points_[:, 0] = grid_points_[:, 0] * w
    grid_points_[:, 1] = grid_points_[:, 1] * h
    scale = 1024 / max(h, w)
    grid_points_ = grid_points_ * scale
    points = all_data[fname]['annotations']['points'].cpu().numpy()
    points = np.concatenate([grid_points_, points], axis=0)
    points = torch.from_numpy(points).cuda()
    # 4. 初始化输出变量
    outputs = {}
    # 5. 循环对每个网格点进行预测
    for indices in torch.arange(len(points)).split(512):
        with torch.no_grad():
            outputs_ = sam.forward_sam_with_embeddings(features, points[indices])

        for k in outputs_:
            if k not in outputs:
                outputs[k] = outputs_[k]
            else:
                outputs[k] = torch.cat([outputs[k], outputs_[k]], dim=0)
    # 预测结果处理
    pred_masks = outputs['pred_logits'].view(-1, 256, 256)  # 重塑预测掩码
    stability_scores = calculate_stability_score(pred_masks, 0, 1.)  # 计算稳定性分数
    pred_ious = outputs['pred_ious'].view(-1)  # 获取预测IoU
    pred_boxes = outputs['pred_boxes'].view(-1, 4)  # 获取预测边界框
    pred_masks = (pred_masks > 0.)  # 将掩码二值化

    # 预测结果筛选
    indices = vision_ops.nms(pred_boxes, pred_ious, 0.7)  # 应用NMS去除重叠框
    mask = torch.zeros(len(pred_ious)).bool().cuda()
    mask[indices] = True
    mask = mask & (stability_scores > 0.95)  # 稳定性阈值过滤
    mask = mask & (pred_ious > 0.88)  # IoU阈值过滤
    bboxes = pred_boxes * max(h, w) / 1024  # 将边界框坐标转换到原始图像尺寸
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, w)  # 限制x坐标在图像范围内
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, h)  # 限制y坐标在图像范围内
    box_areas = vision_ops.box_area(bboxes)  # 计算边界框面积
    mask = mask & ((box_areas < (h * w * 0.75)) & (box_areas > 50.))  # 面积范围过滤
    mask = mask & (((bboxes[:, 3] - bboxes[:, 1]) > 1.) & ((bboxes[:, 2] - bboxes[:, 0]) > 1.))  # 边界框最小尺寸过滤

    pred_masks = pred_masks[mask]
    pred_boxes = pred_boxes[mask]
    pred_ious = pred_ious[mask]

    # 基于pred_masks计算中心点
    contour_center_points = []
    mask = torch.ones(len(pred_masks)).bool()
    contour_center_points = []
    mask = torch.ones(len(pred_masks)).bool()
    for i in range(len(pred_masks)):
        m = pred_masks[i].cpu().numpy().astype(np.uint8)  # 转换掩码为numpy数组
        from ops.foundation_models.segment_anything.utils.amg import remove_small_regions

        # mask = remove_small_regions(mask)
        min_area = 10
        m, changed = remove_small_regions(m, min_area, mode="holes")  # 移除小孔洞
        unchanged = not changed
        m, changed = remove_small_regions(m, min_area, mode="islands")  # 移除小岛屿
        unchanged = unchanged and not changed
        m = m.astype(np.uint8)

        import imutils, cv2

        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            mask[i] = False
            continue
        else:
            cnts = cnts[0] * 4  # 将轮廓坐标放大4倍（对应SAM的下采样比例）
            M = cv2.moments(cnts)  # 计算轮廓矩
            if M['m01'] <= 0.0001:
                mask[i] = False
                continue
            cX = int(M["m10"] / (M["m00"] + 0.0001))  # 计算重心x坐标
            cY = int(M["m01"] / (M["m00"] + 0.0001))  # 计算重心y坐标
            contour_center_points.append([cX, cY])  # 添加到中心点列表

    contour_center_points = torch.Tensor(contour_center_points)
    pred_boxes = pred_boxes[mask]
    pred_ious = pred_ious[mask]
    segment_anything_data[fname] = {
        'pred_boxes': pred_boxes.cpu(),
        'pred_points': contour_center_points.cpu(),
        'pred_ious': pred_ious.cpu(),
    }
    # print(fname)
    # if fname == '6.jpg':
    #     break
    # break

torch.save(segment_anything_data, f'{project_root}/data/fsc147/sam/segment_anything_data_vith.pth')

# 展示SAM预测结果
# read_image(fname)
fname = list(segment_anything_data)[0]
# fname = '2.jpg'
plot_results(read_image(fname),
             points=all_data[fname]['annotations']['points'],
             bboxes=segment_anything_data[fname]['pred_boxes'],
             )

# torch.save(all_data, f'{project_root}/data/fsc147/sam/all_data_vitl.pth')
# torch.save(all_data, f'{project_root}/data/fsc147/sam/all_data_vitr.pth')


# =====================基于真实标注点生成伪标签  =====================

pseudo_boxes_data = {}
for fname in tqdm.tqdm(all_data):
    features = all_data[fname]['features'].cuda()
    points = all_data[fname]['annotations']['points'].cuda()

    outputs = {}
    for indices in torch.arange(len(points)).split(512):
        with torch.no_grad():
            outputs_ = sam.forward_sam_with_embeddings(features, points[indices])

        for k in outputs_:
            if k not in outputs:
                outputs[k] = outputs_[k]
            else:
                outputs[k] = torch.cat([outputs[k], outputs_[k]], dim=0)

    pred_masks = outputs['pred_logits'][torch.arange(len(points)), torch.argmax(outputs['pred_ious'], dim=1)]
    pred_masks = (pred_masks > 0.)
    for i in range(len(pred_masks)):
        m = pred_masks[i].cpu().numpy().astype(np.uint8)
        from ops.foundation_models.segment_anything.utils.amg import remove_small_regions

        min_area = 10
        m, changed = remove_small_regions(m, min_area, mode="holes")
        unchanged = not changed
        m, changed = remove_small_regions(m, min_area, mode="islands")
        unchanged = unchanged and not changed
        m = m.astype(np.uint8)
        pred_masks[i] = torch.from_numpy(m).bool()
    pred_boxes = batched_mask_to_box(
        F.interpolate(pred_masks.unsqueeze(1).float(), size=(1024, 1024)).squeeze(1).bool())
    pseudo_boxes_data[fname] = pred_boxes.cpu()

# 展示伪标签结果
# read_image(fname)
fname = list(pseudo_boxes_data)[0]
# fname = '2.jpg'
plot_results(read_image(fname),
             points=all_data[fname]['annotations']['points'],
             bboxes=pseudo_boxes_data[fname],
             )

torch.save(pseudo_boxes_data, f'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth')

# # Generate text and example CLIP embeddings

# =====================基于clip提取小样本的图像嵌入 =====================
# 加载 CLIP模型
from ops.foundation_models import clip

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


# 使用clip提取少量样本的图像嵌入
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


for fname in tqdm.tqdm(all_data):
    boxes = all_data[fname]['box_examples_coordinates']
    all_data[fname]['example_clip_features'] = extract_clip_features(fname, boxes.reshape(-1, 4)).view(-1,
                                                                                                       boxes.size(0),
                                                                                                       512)
# =====================基于clip提取全量的文本字典=====================

# @title Define hyperparameters
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,

    'temperature': 100.0,
    'use_softmax': False,
}
from easydict import EasyDict

FLAGS = EasyDict(FLAGS)


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace('_', ' ').replace('/', ' or ').lower()
    if rm_dot:
        res = res.rstrip('.')
    return res


single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


def build_text_embedding(categories):
    """
    基于clip文本编码器构建文本嵌入向量

    该函数根据给定的类别列表，使用CLIP模型生成对应的文本嵌入向量。
    支持多种模板和提示工程选项，对文本进行编码和归一化处理。

    Args:
        categories (list): 类别名称列表，用于构建文本嵌入

    Returns:
        dict: 包含类别名称作为键，对应文本嵌入向量作为值的字典
              每个嵌入向量都经过归一化处理并转换为CPU张量
    """
    if FLAGS.prompt_engineering:
        templates = multiple_templates
    else:
        templates = single_template

    with torch.no_grad():
        all_text_embeddings = {}
        print('Building text embeddings...')

        # 遍历每个类别，生成对应的文本嵌入
        for category in tqdm.tqdm(categories):
            # 根据模板格式化文本，处理类别名称并添加冠词
            texts = [
                template.format(processed_name(category, rm_dot=True),
                                article=article(category))
                for template in templates]

            # 根据配置添加"This is"前缀到以冠词开头的文本
            if FLAGS.this_is:
                texts = [
                    'This is ' + text if text.startswith('a') or text.startswith('the') else text
                    for text in texts
                ]

            # 对文本进行tokenize处理并移到GPU
            texts = clip.tokenize(texts)  # tokenize
            texts = texts.cuda()

            # 使用文本编码器生成嵌入向量并进行归一化
            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            # 计算平均嵌入向量并再次归一化
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()

            # 保存处理后的嵌入向量到CPU内存
            all_text_embeddings[category] = text_embedding.float().cpu()

    return all_text_embeddings


# 生成文本嵌入的执行入口
classes = set(all_data[fname]['class_name'] for fname in all_data)
imagenet_classes = classes
print(imagenet_classes)
print(len(imagenet_classes))
zeroshot_weights = build_text_embedding(imagenet_classes)
zeroshot_weights
# 保存文本嵌入向量
torch.save(zeroshot_weights, f'{project_root}/data/fsc147/clip_text_prompt.pth')
