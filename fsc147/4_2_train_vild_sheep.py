"""
此脚本用于训练和评估 VILD（Vision and Language Detection）模型。
该模型结合了视觉特征和语言提示，使用 SAM（Segment Anything Model）和 CLIP 特征进行目标检测和计数任务，
主要应用于 FSC147 数据集上的少样本或零样本检测任务。

输入路径和文件：
- 输入图像：'/home/zzhuang/PseCo/data/fsc147/images_384_VarV2/' 目录下的图像文件
- CLIP 文本提示：'{project_root}/data/fsc147/clip_text_prompt.pth'
- 训练数据：'{project_root}/data/fsc147/sam/all_data_vith.pth'
- 预测数据：'{project_root}/data/fsc147/sam/all_predictions_vith.pth'
- 伪框数据：'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth'

输出路径和文件：
- 模型权重：'{project_root}/data/fsc147/checkpoints/vild.pth'（如果保存的话）
- COCO 格式的评估结果：通过 COCO API 生成的评估指标
- 计数评估指标：MAE, MSE, NAE, SRE 等指标
"""

import sys

sys.path.insert(0, '/mnt/mydisk/wjj/PseCo-main')
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import tqdm
import albumentations as A
import torch.nn as nn
import torchvision.ops as vision_ops
from ops.ops import plot_results, convert_to_cuda

plt.rcParams["figure.dpi"] = 300
torch.cuda.set_device(1)
torch.autograd.set_grad_enabled(False)


# !gpustat
# %%
def read_image(fname):
    img = Image.open(f'/home/zzhuang/PseCo/data/fsc147/images_384_VarV2/{fname}')
    transform = A.Compose([
        A.LongestMaxSize(1024),
        A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    img = Image.fromarray(transform(image=np.array(img))['image'])
    return img


# %%
from ops.foundation_models.segment_anything import build_sam_vit_h

# sam = build_sam_vit_b().cuda().eval()
sam = build_sam_vit_h().cuda().eval()
# %%
project_root = '/mnt/mydisk/wjj/PseCo-main'
clip_text_prompts = torch.load(f'{project_root}/data/fsc147/clip_text_prompt.pth', map_location='cpu')
# %%
all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith.pth', map_location='cpu')
all_predictions = torch.load(f'{project_root}/data/fsc147/sam/all_predictions_vith.pth', map_location='cpu')

all_pseudo_boxes = torch.load(f'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth', map_location='cpu')
# %%
for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    target['image_id'] = fname
    target['predictions'] = all_predictions[fname]
    if all_data[fname]['split'] == 'train':
        target['annotations']['boxes'] = all_pseudo_boxes[fname]['pred_boxes']
        target['annotations']['ious'] = all_pseudo_boxes[fname]['pred_ious']
all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)


# %%
class ROIHead(nn.Module):
    def __init__(self, class_centers):
        super(ROIHead, self).__init__()
        self.image_region_size = 7
        self.linear = nn.Sequential(nn.Linear(256 * self.image_region_size * self.image_region_size, 4096),
                                    nn.ReLU(True), nn.Linear(4096, 512))
        self.bg_head = nn.Parameter(torch.zeros(1, 512))
        self.class_head = class_centers

    def forward(self, features, bboxes):
        features = vision_ops.roi_align(features, bboxes, output_size=(self.image_region_size, self.image_region_size),
                                        spatial_scale=1 / 16.0, aligned=True)
        roi_features = F.normalize(self.linear(features.flatten(1)), dim=1)
        head = torch.cat([self.class_head, F.normalize(self.bg_head, dim=1)], dim=0)
        cls_outs = roi_features.mm(head.T) / 0.01
        return roi_features, cls_outs


# %%
classes = set([all_data[fname]['class_name'] for fname in all_data if all_data[fname]['split'] == 'train'])
classname2idx = {}
for i, classname in enumerate(classes):
    classname2idx[classname] = i
classname2idx
# %%
cls_head = ROIHead(torch.stack([clip_text_prompts[x] for x in classname2idx]).cuda()).cuda().eval()
cls_head(torch.randn(2, 256, 64, 64).cuda(),
         [torch.randint(0, 1024, (2, 4)).float().cuda(), torch.randint(0, 1024, (1, 4)).float().cuda()])
# %%
from ops.loggerx import LoggerX

logger = LoggerX(save_root=None, print_freq=10)
cls_head = ROIHead(torch.stack([clip_text_prompts[x] for x in classname2idx]).cuda()).cuda()
logger.modules = [cls_head, ]
optimizer = torch.optim.AdamW(list(cls_head.parameters()),
                              lr=0.0001, weight_decay=0.0)
acc_grd_step = 1
max_iter = 10000

from ops.grad_scaler import NativeScalerWithGradNormCount

amp = True
scaler = NativeScalerWithGradNormCount(amp=amp)

bs = 32
num_masks = 4
# =======================训练逻辑=========================
for n_iter in range(1, max_iter + 1):
    cls_head.train()

    targets = [all_data[all_image_list['train'][i]] for i in torch.randint(0, len(all_image_list['train']), (bs,))]
    targets = convert_to_cuda(targets)
    features = torch.cat([t['features'] for t in targets])
    num_anchors = 256
    pos_ratios = 0.25
    anchor_boxes = []
    query_features, query_labels = [], []

    for t in targets:

        fname = t['image_id']

        annotations = t['annotations']
        gt_bboxes = annotations['boxes']
        gt_points = annotations['points']

        min_scores = 0.05
        max_points = 1000

        pred_points_score = all_data[fname]['predictions']['pred_points_score']
        mask = torch.zeros(pred_points_score.size(0))
        mask[:min(pred_points_score.size(0), max_points)] = 1
        mask[pred_points_score < min_scores] = 0
        candidate_boxes = all_data[fname]['predictions']['pred_boxes'][mask.bool()].cuda()[:, :num_masks]

        iou_scores = vision_ops.box_iou(candidate_boxes.reshape(-1, 4), gt_bboxes)
        iou_scores = iou_scores.max(dim=1).values.reshape(-1, num_masks)

        anchor_indices = torch.randint(0, candidate_boxes.size(0), (num_anchors,))
        pos_mask = (iou_scores.max(1).values > 0.5).float()
        if pos_mask.sum() > 0:
            pos_indices = torch.multinomial(pos_mask, int(min(num_anchors * pos_ratios, pos_mask.sum())))
            anchor_indices[:len(pos_indices)] = pos_indices

        cur_labels = torch.ones(len(anchor_indices), num_masks).cuda() * len(classname2idx)
        cur_labels[iou_scores[anchor_indices] > 0.5] = classname2idx[t['class_name']]
        query_labels.append(cur_labels.view(-1))
        anchor_boxes.append(candidate_boxes[anchor_indices].view(-1, 4))
    query_labels = torch.cat(query_labels, dim=0)

    targets_a = [all_data[all_image_list['all'][i]] for i in torch.randint(0, len(all_image_list['all']), (bs,))]
    targets_a = convert_to_cuda(targets_a)
    features_a = torch.cat([t['features'] for t in targets_a])
    clip_boxes, clip_target_features, = [], []
    for t in targets_a:
        fname = t['image_id']

        region_boxes = all_data[fname]['predictions']['clip_regions']['boxes'].float().cuda()[:, :num_masks]
        rand_indices = torch.randint(0, len(region_boxes), (16,))
        clip_boxes.append(region_boxes[rand_indices].cuda().view(-1, 4))
        clip_target_features += [x[0].cuda() for x in
                                 all_data[fname]['predictions']['clip_regions']['clip_embeddings'][:, :num_masks][
                                     rand_indices].split(1, dim=0)]
    clip_target_features = torch.cat(clip_target_features)

    with torch.autograd.set_grad_enabled(True) and torch.autocast(device_type='cuda', enabled=amp):

        cls_embs, cls_outs = cls_head(features, anchor_boxes)
        cls_loss = F.cross_entropy(cls_outs, query_labels.long())

        cls_embs2, cls_outs2 = cls_head(features_a, clip_boxes)
        kd_loss = F.l1_loss(cls_embs2, clip_target_features, reduction='none').sum(1).mean()

        loss = cls_loss + kd_loss * 0.5

        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        scaler(loss, optimizer=optimizer, update_grad=update_params)

    logger.msg([cls_loss, kd_loss], n_iter)
# =========================测试部分=======================
# %%
# cls_head = ROIHead(torch.stack([clip_text_prompts[x] for x in classname2idx]).cuda()).cuda()
# cls_head.load_state_dict(torch.load(f'{project_root}/data/fsc147/checkpoints/vild.pth', map_location='cpu'))
# torch.save(cls_head.state_dict(), f'{project_root}/data/fsc147/checkpoints/vild.pth')
# %%
# fname = '5.jpg'
fname = '6860.jpg'
# fname = '1123.jpg'
# fname = '2858.jpg'
# fname = '7611.jpg'
# fname = '3337.jpg'
# fname = all_image_list['test'][3]
# fname = '5514.jpg'
# fname = '3309.jpg'
# fname = '3308.jpg'
# fname = '5513.jpg'
_ = cls_head.eval()
# %%
annotations = all_data[fname]['annotations']
gt_points = annotations['points']
# %%
with torch.no_grad():
    # example_features = clip_text_prompts[all_data[fname]['class_name']].unsqueeze(0).cuda()
    example_features = all_data[fname]['example_clip_features'].cuda()
    min_scores = 0.05
    max_points = 1000
    pred_points_score = all_data[fname]['predictions']['pred_points_score']
    mask = torch.zeros(pred_points_score.size(0))
    mask[:min(pred_points_score.size(0), max_points)] = 1
    mask[pred_points_score < min_scores] = 0
    pred_boxes = all_data[fname]['predictions']['pred_boxes'][mask.bool()].cuda()
    pred_ious = all_data[fname]['predictions']['pred_ious'][mask.bool()].cuda()
    cls_outs = []
    for indices in torch.arange(len(pred_boxes)).split(128):
        cls_outs_ = cls_head(all_data[fname]['features'].cuda(), [pred_boxes[indices, :num_masks].reshape(-1, 4), ])
        cls_outs.append(cls_outs_[0].mm(example_features.T).mean(1).view(-1, num_masks))
    cls_outs = torch.cat(cls_outs)
    pred_boxes = pred_boxes[torch.arange(len(pred_boxes)), torch.argmax(cls_outs, dim=1)]
    scores = cls_outs.max(1).values
    indices = vision_ops.nms(pred_boxes, scores, 0.5)
    pred_boxes = pred_boxes[indices]
    scores = scores[indices]
# %%
plot_results(read_image(fname),
             bboxes=pred_boxes[scores > 0.0],
             #  bboxes=pred_boxes,
             # bboxes=gt_boxes,
             # bboxes=all_data[fname]['box_examples_coordinates'],
             )
# %%
# ========== 替换 detectron2 的核心部分 ==========
# 1. 导入 pycocotools 替代 detectron2 的 COCOEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# 2. 加载 COCO 标注文件
# coco_gt = COCO(f"{project_root}/data/fsc147/instances_test_val_bin.json")
coco_gt = COCO(f"/mnt/mydisk/wjj/dataset/FSC_147/instances_test_val_bin.json")

# 3. 定义评估用的 split
split = 'test'
# split = 'val'
image_list = [fname for fname in all_data if all_data[fname]['split'] == split]

# 4. 准备预测结果（COCO 格式）
coco_preds = []
all_predictions = {}

for fname in tqdm.tqdm(image_list):
    features = all_data[fname]['features'].cuda()
    with torch.no_grad():
        cls_head.eval()
        # few shot
        # example_features = all_data[fname]['example_clip_features'].cuda()
        # zero shot
        class_name = all_data[fname]['class_name']
        example_features = clip_text_prompts[class_name].unsqueeze(0).cuda()

    min_scores = 0.05
    max_points = 1000
    pred_points_score = all_data[fname]['predictions']['pred_points_score']
    mask = torch.zeros(pred_points_score.size(0))
    mask[:min(pred_points_score.size(0), max_points)] = 1
    mask[pred_points_score < min_scores] = 0
    pred_boxes = all_data[fname]['predictions']['pred_boxes'][:, :num_masks][mask.bool()].cuda()
    pred_ious = all_data[fname]['predictions']['pred_ious'][:, :num_masks][mask.bool()].cuda()

    all_pred_boxes = []
    all_pred_scores = []
    for indices in torch.arange(len(pred_boxes)).split(128):
        with torch.no_grad():
            cls_outs_ = cls_head(all_data[fname]['features'].cuda(), [pred_boxes[indices].reshape(-1, 4), ])
            pred_logits = cls_outs_[0].mm(example_features.T).mean(1).view(-1, num_masks)

            pred_logits = pred_logits * pred_ious[indices]

            all_pred_boxes.append(pred_boxes[indices, torch.argmax(pred_logits, dim=1)])
            all_pred_scores.append(pred_logits.max(dim=1).values)

    height, width = all_data[fname]['height'], all_data[fname]['width']
    scale = max(height, width) / 1024.
    pred_boxes = torch.cat(all_pred_boxes) * scale
    pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]].clamp(0, width)
    pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]].clamp(0, height)
    pred_scores = torch.cat(all_pred_scores)
    box_area = vision_ops.box_area(pred_boxes)
    mask = (box_area < (height * width * 0.75)) & (box_area > 10)
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]

    nms_indices = vision_ops.nms(pred_boxes, pred_scores, 0.5)
    pred_boxes = pred_boxes[nms_indices]
    pred_scores = pred_scores[nms_indices]

    # 保存预测结果供后续计数评估
    image_id = int(fname[:-4])
    all_predictions[fname] = {
        'boxes': pred_boxes.cpu().numpy(),
        'scores': pred_scores.cpu().numpy(),
        'image_id': image_id
    }

    # 转换为 COCO 预测格式
    for box, score in zip(pred_boxes.cpu().numpy(), pred_scores.cpu().numpy()):
        # COCO 格式要求: [x1, y1, w, h] (detectron2 用的是 [x1, y1, x2, y2])
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        coco_preds.append({
            "image_id": image_id,
            "category_id": 1,  # 假设只有一个类别，根据你的实际数据调整
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score)
        })

# 5. 运行 COCO 评估
# 将预测结果加载到 COCO API
coco_dt = coco_gt.loadRes(coco_preds)

# 创建评估器并运行评估
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
# 只评估指定的图片
coco_eval.params.imgIds = [int(x[:-4]) for x in image_list]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 6. 计算计数相关指标 (MAE, MSE, NAE, SRE)
mae, mse, nae, sre = [], [], [], []
thresholds = np.arange(0, 1., 0.01)
for thresh in thresholds:
    total_mae = 0.
    total_mse = 0.
    total_nae = 0.
    total_sre = 0.
    for i, fname in enumerate(image_list):
        num_points = len(all_data[fname]['annotations']['points'])
        pred_count = (all_predictions[fname]['scores'] > thresh).sum()
        err = abs(num_points - pred_count)
        total_mae += err
        total_mse += err ** 2
        total_nae += err / num_points
        total_sre += err ** 2 / num_points
    cnt = len(image_list)
    mae.append(float(total_mae / cnt))
    mse.append(float((total_mse / cnt) ** 0.5))
    nae.append(float(total_nae / cnt))
    sre.append(float((total_sre / cnt) ** 0.5))

# 输出最优阈值下的指标
best_idx = np.argmin(mae)
print(f"最优阈值: {thresholds[best_idx]:.2f}")
print(f"MAE: {mae[best_idx]:.4f}")
print(f"MSE: {mse[best_idx]:.4f}")
print(f"NAE: {nae[best_idx]:.4f}")
print(f"SRE: {sre[best_idx]:.4f}")

# 7. 指定阈值计算指标
thresh = 0.31
total_mae = 0.
total_mse = 0.
total_nae = 0.
total_sre = 0.
for i, fname in enumerate(image_list):
    num_points = len(all_data[fname]['annotations']['points'])
    pred_count = (all_predictions[fname]['scores'] > thresh).sum()
    err = abs(num_points - pred_count)
    total_mae += err
    total_mse += err ** 2
    total_nae += err / num_points
    total_sre += err ** 2 / num_points
cnt = len(image_list)

final_mae = float(total_mae / cnt)
final_mse = float((total_mse / cnt) ** 0.5)
final_nae = float(total_nae / cnt)
final_sre = float((total_sre / cnt) ** 0.5)

print(f"\n指定阈值 {thresh} 下的指标:")
print(f"MAE: {final_mae:.4f}")
print(f"MSE: {final_mse:.4f}")
print(f"NAE: {final_nae:.4f}")
print(f"SRE: {final_sre:.4f}")