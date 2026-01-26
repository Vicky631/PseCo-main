"""
此脚本用于训练ROI Head模型，专门用于目标检测和计数任务。
该模型结合了视觉特征和语言提示，使用SAM（Segment Anything Model）和CLIP特征进行少样本或零样本检测任务，
主要应用于FSC147数据集上的目标检测与计数。

输入路径和文件：
- 输入图像：'/mnt/mydisk/wjj/dataset/FSC_147/images_384_VarV2/' 目录下的图像文件
- COCO标注文件：'/mnt/mydisk/wjj/dataset/FSC_147/annotation_FSC147_384_with_gt.json'
- CLIP文本提示：'{project_root}/data/fsc147/clip_text_prompt.pth'
- 训练数据：'{project_root}/data/fsc147/sam/all_data_vith.pth'
- 预测数据：'{project_root}/data/fsc147/sam/all_predictions_vith.pth'
- 伪框数据：'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth'

输出路径和文件：
- 模型权重：'{project_root}/data/fsc147/checkpoints/cls_head/ckpt/{run_name}'
- COCO格式的评估结果：通过COCO API生成的评估指标
- 计数评估指标：MAE, RMSE, NAE, SRE等指标
"""
import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_module = logging.getLogger(__name__)

project_root = '/ZHANGyong/wjj/PseCo-main'
dataset_root = '/ZHANGyong/wjj/dataset'
online_models = '/ZHANGyong/wjj/online_models/torch_cache/hub/checkpoints'
sys.path.insert(0, project_root)
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
from ops.ops import _nms, plot_results, convert_to_cuda

# 命令行参数定义
import argparse

logging.info("开始解析命令行参数")
parser = argparse.ArgumentParser('Default arguments for training of different methods')
parser.add_argument('--wandb', help='wandb', action='store_true')
parser.add_argument('--zeroshot', help='zeroshot', action='store_true')
parser.add_argument('--arch', help='arch: vitb, vitl, vith', type=str, default='vith')
parser.add_argument('--entity', help='wandb user name', type=str, default='zzhuang')
opts = parser.parse_args()
print(opts)
logging.info(f"命令行参数解析完成: {opts}")
# 数据集注册和初始化================================================
# from detectron2.data.datasets import register_coco_instances
#
# register_coco_instances("fsc_test_val", {}, f"{project_root}/data/fsc147/instances_test_val_bin.json",
#                         f"{project_root}/data/fsc147/images_384_VarV2")
import json
from pycocotools.coco import COCO

# 加载COCO标注文件（原代码中register_coco_instances的作用仅为评估，此处直接加载）
# coco_gt = COCO(f"{project_root}/data/fsc147/instances_test_val_bin.json")
logging.info("开始加载COCO标注文件")
coco_gt = COCO(f"{dataset_root}/FSC_147/instances_test_val_bin.json")
logging.info("COCO标注文件加载完成")
# ============================================================
torch.autograd.set_grad_enabled(False)

from ops.foundation_models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, \
    build_sam, build_sam_vit_b, build_sam_vit_h

# 模型加载
logging.info("开始加载模型和数据")
# sam = build_sam_vit_b().cuda().eval()
# all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vitb.pth', map_location='cpu')
# all_predictions = torch.load(f'{project_root}/data/fsc147/sam/all_predictions_vitb.pth', map_location='cpu')

# load all features and proposals
if opts.arch == 'vith':
    logging.info("加载ViT-H架构模型和数据")
    sam = build_sam_vit_h(f"{online_models}/sam_vit_h_4b8939.pth").cuda().eval()
    logging.info("开始加载all_data")
    all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith_v5_fix.pth', map_location='cpu')
    logging.info("开始加载all_predictions")
    all_predictions = torch.load(f'{project_root}/data/fsc147/sam/all_predictions_vith.pth', map_location='cpu')
    logging.info("模型和数据加载完成")
else:
    raise NotImplementedError

# 加载CLIP文本提示和伪标签
logging.info("开始加载CLIP文本提示")
clip_text_prompts = torch.load(f'{project_root}/data/fsc147/clip_text_prompt.pth', map_location='cpu')
logging.info("开始加载伪标签")
all_pseudo_boxes = torch.load(f'{project_root}/data/fsc147/sam/pseudo_boxes_data_vith.pth', map_location='cpu')
# logging.info("开始更新数据集")
# for fname in tqdm.tqdm(all_data):
#     target = all_data[fname]
#     target['image_id'] = fname
#     target['predictions'] = all_predictions[fname]
#     if all_data[fname]['split'] == 'train':
#         target['annotations']['boxes'] = all_pseudo_boxes[fname]['pred_boxes']
#         target['annotations']['ious'] = all_pseudo_boxes[fname]['pred_ious']
# logging.info("数据集更新完成")
logging.info("开始更新数据集")
logging.info(f"伪框数据总数量: {len(all_pseudo_boxes)}")  # 打印总数据量

# 遍历数据集并添加详细日志
for idx, fname in enumerate(tqdm.tqdm(all_data)):
    try:
        target = all_data[fname]
        target['image_id'] = fname
        target['predictions'] = all_predictions[fname]

        # 打印当前处理的文件信息
        logging.info(f"正在处理第 {idx + 1}/{len(all_data)} 个文件: {fname}")

        if all_data[fname]['split'] == 'train':
            # 先检查该文件是否存在于伪框数据中
            if fname not in all_pseudo_boxes:
                logging.error(f"文件 {fname} 不存在于伪框数据中！")
                continue

            # 打印 pred_boxes 的关键信息（维度、形状、类型）
            pred_boxes = all_pseudo_boxes[fname]
            # pred_boxes = all_pseudo_boxes[fname]['pred_boxes']
            # pred_ious = all_pseudo_boxes[fname]['pred_ious']

            logging.info(
                f"文件 {fname} - pred_boxes 维度: {pred_boxes.dim()}, 形状: {pred_boxes.shape}, 类型: {pred_boxes.dtype}")
            # logging.info(
            #     f"文件 {fname} - pred_ious 维度: {pred_ious.dim()}, 形状: {pred_ious.shape}, 类型: {pred_ious.dtype}")

            # 赋值（你的原有逻辑）
            target['annotations']['boxes'] = pred_boxes
            # target['annotations']['ious'] = pred_ious

    except IndexError as e:
        # 捕获索引错误并打印详细信息
        logging.error(f"处理文件 {fname} 时发生索引错误: {str(e)}")
        logging.error(f"出错文件 {fname} 的 pred_boxes 详细信息: {all_pseudo_boxes[fname]['pred_boxes']}")
        raise  # 可选：如果想终止程序，保留raise；如果想继续处理其他文件，注释掉raise
    except Exception as e:
        # 捕获其他所有异常
        logging.error(f"处理文件 {fname} 时发生未知错误: {str(e)}", exc_info=True)
        raise  # 同上，可选是否终止

logging.info("数据集更新完成")
#数据集划分
logging.info("开始划分数据集")
all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)
logging.info(f"数据集划分完成: 训练集 {len(all_image_list['train'])} 张, 验证集 {len(all_image_list['val'])} 张, 测试集 {len(all_image_list['test'])} 张")

from models import ROIHeadMLP as ROIHead

num_masks = 5
run_name = 'MLP_small_box_w1'
if opts.zeroshot:
    run_name += '_zeroshot'
cls_loss2_weight = 1.0

# 训练器初始化
logging.info("开始初始化训练器")
from ops.loggerx import LoggerX

logger = LoggerX(save_root=f'{project_root}/data/fsc147/checkpoints/cls_head/ckpt/{run_name}',
                 # print_freq=10,
                 name=run_name,
                 enable_wandb=opts.wandb,
                 config=opts,
                 entity=opts.entity,
                 project='Counting')
cls_head = ROIHead().cuda()
logger.modules = [cls_head, ]
optimizer = torch.optim.AdamW(list(cls_head.parameters()), lr=0.0001, weight_decay=0.00001)
acc_grd_step = 1
max_iter = 10000
bs = 32
logging.info(f"训练器初始化完成，最大迭代次数: {max_iter}, 批大小: {bs}")
# 混合精度训练设置
logging.info("开始设置混合精度训练")
from ops.grad_scaler import NativeScalerWithGradNormCount

amp = True
scaler = NativeScalerWithGradNormCount(amp=amp)
logging.info("混合精度训练设置完成")


def evaluate(split, results, threshold=None):
    """
    评估模型在指定数据集上的性能

    Args:
        split (str): 数据集划分，如 'train', 'val', 'test'
        results (dict): 存储评估结果的字典
        threshold (float, optional): 用于计数评估的阈值，默认为None

    Returns:
        dict: 包含评估指标的字典，包括bbox指标和计数指标(MAE, RMSE, NAE, SRE)
    """
    logging.info(f"开始评估模型在 {split} 集上的性能")
    image_list = [fname for fname in all_data if all_data[fname]['split'] == split]
    logging.info(f"{split} 集图像数量: {len(image_list)}")
    # =====================================================================
    # from detectron2.evaluation import COCOEvaluator
    #
    # coco_evaluator = COCOEvaluator(dataset_name=f'fsc_test_val', tasks=['bbox', ],
    #                                output_dir=f'{project_root}/data/temp', max_dets_per_image=1000)
    # coco_evaluator.reset()
    #
    # from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
    #
    # all_predictions = {}
    # 替换detectron2的COCOEvaluator：使用原生pycocotools评估
    from pycocotools.cocoeval import COCOeval

    # 准备COCO评估的预测结果列表
    coco_preds = []
    all_predictions = {}
    # ==============================================

    logging.info(f"开始处理 {split} 集中的图像")
    for fname in tqdm.tqdm(image_list):
        features = all_data[fname]['features'].cuda()
        with torch.no_grad():
            cls_head.eval()
            if opts.zeroshot:
                example_features = clip_text_prompts[all_data[fname]['class_name']].unsqueeze(0).cuda()
            else:
                example_features = all_data[fname]['example_clip_features'].cuda()

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
                cls_outs_ = cls_head(features, [pred_boxes[indices], ], [example_features, ] * len(indices))
                pred_logits = cls_outs_.sigmoid().view(-1, len(example_features), num_masks).mean(1)
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
        # ======================================
        # nms_indices = vision_ops.nms(pred_boxes, pred_scores, 0.5)
        # instances = Instances((height, width))
        # pred_boxes = pred_boxes[nms_indices]
        # pred_scores = pred_scores[nms_indices]
        # instances.pred_boxes = Boxes(pred_boxes)
        # instances.scores = pred_scores
        # instances.pred_classes = torch.zeros(len(pred_boxes)).cuda().long()
        # prediction = {"image_id": int(fname[:-4]), "instances": instances}
        # coco_evaluator.process(
        #     [{'file_name': fname, 'height': height, 'width': width, 'image_id': int(fname[:-4])}],
        #     [prediction, ])
        # all_predictions[fname] = prediction
        # 替换detectron2的Instances/Boxes：原生张量+COCO格式转换
        nms_indices = vision_ops.nms(pred_boxes, pred_scores, 0.5)
        pred_boxes = pred_boxes[nms_indices]
        pred_scores = pred_scores[nms_indices]

        if torch.rand(1).item() < 0.01:  # 每100个图像打印一次进度，避免日志过多
            logging.info(f"评估过程中: 处理图像 {fname}, 预测框数量: {len(pred_boxes)}, 预测分数范围: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")

        # 转换为COCO评估格式（x1,y1,w,h）：原代码是x1,y1,x2,y2，需要转换
        image_id = int(fname[:-4])
        for box, score in zip(pred_boxes.cpu().numpy(), pred_scores.cpu().numpy()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            coco_preds.append({
                "image_id": image_id,
                "category_id": 0,  # 类别ID和原代码保持一致（0类）
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

        # 保存预测结果（用于计数评估）
        all_predictions[fname] = {
            "image_id": image_id,
            "boxes": pred_boxes,
            "scores": pred_scores
        }

        # break
    # detection_results = coco_evaluator.evaluate([int(x[:-4]) for x in image_list])
    # for k in detection_results['bbox']:
    #     results[split + '_' + k] = detection_results['bbox'][k]
    # 运行原生COCO评估（替代detectron2的coco_evaluator.evaluate）
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # 筛选需要评估的图片ID
    eval_img_ids = [int(x[:-4]) for x in image_list]
    coco_eval.params.imgIds = eval_img_ids
    # 执行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取评估指标（和detectron2输出格式对齐）
    detection_results = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'APs': coco_eval.stats[3],
        'APm': coco_eval.stats[4],
        'APl': coco_eval.stats[5],
        'AR1': coco_eval.stats[6],
        'AR10': coco_eval.stats[7],
        'AR100': coco_eval.stats[8],
        'ARs': coco_eval.stats[9],
        'ARm': coco_eval.stats[10],
        'ARl': coco_eval.stats[11]
    }
    for k in detection_results:
        results[split + '_' + k] = detection_results[k]

    def eval_counting(thresh):
        """
        计算计数评估指标

        Args:
            thresh (float): 用于判断目标的阈值

        Returns:
            tuple: (mae, mse, nae, sre) - 平均绝对误差、均方根误差、归一化平均误差、平方相对误差
        """
        total_mae = 0.
        total_mse = 0.
        total_nae = 0.
        total_sre = 0.
        for i, fname in enumerate(image_list):
            num_points = len(all_data[fname]['annotations']['points'])
            # err = abs(num_points - (all_predictions[fname]['instances'].scores > thresh).sum())
            # 替换detectron2的instances.scores：直接访问保存的scores张量
            pred_cnt = (all_predictions[fname]['scores'] > thresh).sum().item()
            err = abs(num_points - pred_cnt)
            total_mae += err
            total_mse += err ** 2
            total_nae += err / num_points
            total_sre += err ** 2 / num_points
        cnt = len(image_list)
        mae = float(total_mae / cnt)
        mse = float((total_mse / cnt) ** 0.5)
        nae = float(total_nae / cnt)
        sre = float((total_sre / cnt) ** 0.5)
        return mae, mse, nae, sre

    if threshold is None:
        mae, mse, nae, sre = [], [], [], []
        thresholds = np.arange(0, 1., 0.01)
        for thresh in thresholds:
            mae_, mse_, nae_, sre_ = eval_counting(thresh)
            mae.append(mae_)
            mse.append(mse_)
            nae.append(nae_)
            sre.append(sre_)
        # mae, mse, nae, sre, thresh = mae[np.argmin(mae)], mse[np.argmin(mae)], nae[np.argmin(mae)], sre[np.argmin(mae)], \
        #     thresholds[np.argmin(mae)]
        best_idx = np.argmin(mae)
        mae, mse, nae, sre, thresh = mae[best_idx], mse[best_idx], nae[best_idx], sre[best_idx], thresholds[best_idx]
        results['THRESH'] = thresh
    else:
        mae, mse, nae, sre = eval_counting(threshold)
    results[split + '_MAE'] = mae
    results[split + '_RMSE'] = mse
    results[split + '_NAE'] = nae
    results[split + '_SRE'] = sre
    return results


# 主训练循环：迭代训练模型，每1000次迭代进行一次验证和测试评估
logging.info("开始主训练循环")
for n_iter in range(1, max_iter + 1):
    logging.info(f"当前迭代次数: {n_iter}/{max_iter}")
    cls_head.train()

    logging.debug(f"第 {n_iter} 次迭代: 开始准备训练数据")
    targets = [all_data[all_image_list['train'][i]] for i in torch.randint(0, len(all_image_list['train']), (bs,))]
    targets = convert_to_cuda(targets)
    features = torch.cat([t['features'] for t in targets])
    logging.debug(f"第 {n_iter} 次迭代: 训练数据准备完成")
    num_anchors = 256
    pos_ratios = 0.25
    anchor_boxes = []
    query_features, query_labels = [], []
    # 训练阶段
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

        cur_labels = torch.zeros(len(anchor_indices), num_masks).cuda()
        cur_labels[iou_scores[anchor_indices] > 0.5] = 1.
        query_labels.append(cur_labels)
        anchor_boxes.append(candidate_boxes[anchor_indices])
        if opts.zeroshot:
            example_features = clip_text_prompts[t['class_name']].cuda().unsqueeze(0)
        else:
            example_features = all_data[fname]['example_clip_features'][
                torch.randint(0, len(all_data[fname]['example_clip_features']), (1,))].cuda()
        query_features += [example_features, ] * len(anchor_indices)
    query_labels = torch.cat(query_labels, dim=0)

    logging.debug(f"第 {n_iter} 次迭代: 开始准备额外训练数据")
    targets_a = [all_data[all_image_list['all'][i]] for i in torch.randint(0, len(all_image_list['all']), (bs,))]
    targets_a = convert_to_cuda(targets_a)
    features_a = torch.cat([t['features'] for t in targets_a])
    logging.debug(f"第 {n_iter} 次迭代: 额外训练数据准备完成")
    clip_boxes, clip_target_features, clip_query_labels = [], [], []
    for t in targets_a:
        fname = t['image_id']

        region_boxes = all_data[fname]['predictions']['clip_regions']['boxes'].float().cuda()[:, :num_masks]
        rand_indices = torch.randint(0, len(region_boxes), (min(16, len(region_boxes)),))
        for i in rand_indices:
            iou_scores = vision_ops.box_iou(region_boxes[i], region_boxes[i])
            cur_labels = torch.zeros_like(iou_scores)
            cur_labels[iou_scores > 0.5] = 1.
            clip_query_labels.append(cur_labels)
        clip_boxes.append(region_boxes[rand_indices].cuda())
        clip_target_features += [x[0].cuda() for x in
                                 all_data[fname]['predictions']['clip_regions']['clip_embeddings'][:, :num_masks][
                                     rand_indices].split(1, dim=0)]
    clip_query_labels = torch.cat(clip_query_labels)

    # 计算损失
    logging.debug(f"第 {n_iter} 次迭代: 开始计算损失")
    with torch.autograd.set_grad_enabled(True) and torch.autocast(device_type='cuda', enabled=amp):

        # cls_outs = cls_head(features, anchor_boxes, query_features)
        # 修改成：
        print(f"调用cls_head前：")
        print(f"query_features（即prompts）长度：{len(query_features)}")
        print(f"异常位置5120的query_features形状：{query_features[5120].shape}")  # 替换成你找到的异常位置
        cls_outs = cls_head(features, anchor_boxes, query_features)
        cls_loss = F.binary_cross_entropy_with_logits(cls_outs, query_labels, reduction='none')
        loss_mask = (query_labels >= 0).float()
        cls_loss = (cls_loss * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        cls_outs2 = cls_head(features_a, clip_boxes, clip_target_features)
        cls_loss2 = F.binary_cross_entropy_with_logits(cls_outs2, clip_query_labels, reduction='none')
        loss_mask = (clip_query_labels >= 0).float()
        cls_loss2 = (cls_loss2 * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        loss = cls_loss + cls_loss2 * cls_loss2_weight
        logging.debug(f"第 {n_iter} 次迭代: 损失计算完成 - 总损失: {loss.item():.4f}, 分类损失: {cls_loss.item():.4f}, 分类损失2: {cls_loss2.item():.4f}")

        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        # 反向传播
        scaler(loss, optimizer=optimizer, update_grad=update_params)
    logging.debug(f"第 {n_iter} 次迭代: 损失计算和反向传播完成")

    batch_pos_ratio = (query_labels == 1).sum() / ((query_labels == 1).sum() + (query_labels == 0).sum())
    batch_pos_ratio2 = (clip_query_labels == 1).sum() / (
            (clip_query_labels == 1).sum() + (clip_query_labels == 0).sum())
    logging.debug(f"第 {n_iter} 次迭代: 正样本比例 - 训练批次: {batch_pos_ratio:.4f}, 额外批次: {batch_pos_ratio2:.4f}")
    logger.msg([cls_loss, cls_loss2, batch_pos_ratio, batch_pos_ratio2], n_iter)

    if n_iter % 1000 == 0:
        logging.info(f"第 {n_iter} 次迭代: 开始评估和保存检查点")
        results = {}
        # set the threshold at val set
        evaluate('val', results)
        evaluate('test', results, results['THRESH'])
        logger.checkpoints(n_iter)
        logger.msg(results, n_iter)
        logging.info(f"第 {n_iter} 次迭代: 评估和保存检查点完成，结果: {results}")

