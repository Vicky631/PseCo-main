#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PseCo推理测试脚本
支持推理任意数据集，替换数据集时仅需修改配置文件/命令行参数，无需改代码

功能：
1. 支持推理任意数据集（通过配置文件或命令行参数指定）
2. 自动适配不同标注格式（COCO/自定义格式）
3. 输出推理结果和评估指标（MAE/RMSE等）
"""
import argparse
import sys
import os
import logging
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.path_manager import PathManager
from utils.annotation_converter import AnnotationConverter


def setup_logging(log_file: str = None):
    """设置日志系统"""
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger, log_file


def load_model_checkpoints(config, ckpt_path: str = None):
    """
    加载模型权重

    Args:
        config: 配置对象
        ckpt_path: 模型权重路径（如果为None，则从配置中获取）

    Returns:
        dict: 包含所有模型权重的字典
    """
    logger = logging.getLogger(__name__)
    paths = PathManager(config)

    checkpoints = {}

    # 加载PointDecoder（阶段2的输出）
    if ckpt_path:
        point_decoder_path = Path(ckpt_path)
    else:
        point_decoder_path = paths.get_stage2_output_point_decoder()

    if point_decoder_path.exists():
        logger.info(f"加载PointDecoder权重: {point_decoder_path}")
        checkpoints['point_decoder'] = torch.load(point_decoder_path, map_location='cpu')
    else:
        raise FileNotFoundError(f"PointDecoder权重文件不存在: {point_decoder_path}")

    # 加载ROIHead（阶段4的输出）
    run_name = config.get('training.stage4.run_name', 'MLP_small_box_w1')
    if config.get('training.stage4.mode') == 'zeroshot':
        run_name += '_zeroshot'

    roi_head_dir = paths.get_stage4_output_dir(run_name)
    if roi_head_dir.exists():
        # 查找最新的checkpoint
        ckpt_files = list(roi_head_dir.glob('*.pth'))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"加载ROIHead权重: {latest_ckpt}")
            checkpoints['roi_head'] = torch.load(latest_ckpt, map_location='cpu')
        else:
            logger.warning(f"ROIHead目录存在但无checkpoint文件: {roi_head_dir}")
    else:
        logger.warning(f"ROIHead目录不存在: {roi_head_dir}")

    return checkpoints


def load_test_data(test_data_root: str, anno_path: str, dataset_type: str = 'auto'):
    """
    加载测试数据

    Args:
        test_data_root: 测试数据根目录
        anno_path: 标注文件路径
        dataset_type: 数据集类型（auto/coco/fsc147_json/custom_json）

    Returns:
        dict: 测试数据字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载测试数据: {test_data_root}")
    logger.info(f"标注文件: {anno_path}")
    logger.info(f"数据集类型: {dataset_type}")

    # 检测格式
    if dataset_type == 'auto':
        dataset_type = AnnotationConverter.detect_format(anno_path)
        logger.info(f"自动检测到格式: {dataset_type}")

    # 加载标注
    if dataset_type == 'coco':
        from pycocotools.coco import COCO
        coco_gt = COCO(anno_path)
        # 转换为内部格式
        test_data = {}
        for img_id in coco_gt.getImgIds():
            img_info = coco_gt.loadImgs(img_id)[0]
            fname = img_info['file_name']
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            # 提取points（从bbox中心点）
            points = []
            for ann in anns:
                bbox = ann['bbox']  # [x, y, w, h]
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                points.append([center_x, center_y])

            test_data[fname] = {
                'width': img_info['width'],
                'height': img_info['height'],
                'points': points,
                'image_id': img_id
            }
    elif dataset_type in ['fsc147_json', 'custom_json']:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_data = {}
        for fname, target in data.items():
            # 提取points
            if 'annotations' in target:
                if isinstance(target['annotations'], dict):
                    points = [ann.get('points', []) for ann in target['annotations'].values()]
                else:
                    points = target['annotations']
            elif 'points' in target:
                points = target['points']
            else:
                points = []

            test_data[fname] = {
                'width': target.get('width', 0),
                'height': target.get('height', 0),
                'points': points,
                'image_id': fname
            }
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    logger.info(f"✅ 加载完成，共 {len(test_data)} 张图像")
    return test_data, dataset_type


def run_inference(
        config,
        test_data: Dict,
        test_data_root: str,
        checkpoints: Dict,
        device: str = 'cuda:0',
        mode: str = 'zeroshot'  # 'fewshot' or 'zeroshot'
):
    """
    运行推理（参考4_1_train_roi_head.py和4_2_train_vild.py的推理逻辑）

    Args:
        config: 配置对象
        test_data: 测试数据字典
        test_data_root: 测试数据根目录
        checkpoints: 模型权重字典
        device: 设备
        mode: 推理模式 ('fewshot' 或 'zeroshot')

    Returns:
        dict: 推理结果 {图像名: {预测框, 预测分数, 预测计数}}
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始推理")
    logger.info("=" * 60)
    logger.info(f"推理模式: {mode}")

    # 加载ROIHead模型
    from models import ROIHeadMLP as ROIHead
    import torchvision.ops as vision_ops
    import torch.nn.functional as F

    logger.info("加载ROIHead模型")
    cls_head = ROIHead().to(device).eval()
    if 'roi_head' in checkpoints:
        state_dict = checkpoints['roi_head']
        # 尝试不同的state_dict格式
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                cls_head.load_state_dict(state_dict['model'])
            elif 'cls_head' in state_dict:
                cls_head.load_state_dict(state_dict['cls_head'])
            else:
                # 直接是模型权重
                cls_head.load_state_dict(state_dict)
        else:
            cls_head.load_state_dict(state_dict)
        logger.info("✅ ROIHead权重加载成功")
    else:
        logger.warning("⚠️ 未找到ROIHead权重，使用未训练的模型")

    # 加载CLIP文本提示（用于zero-shot）
    paths = PathManager(config)
    clip_text_prompts = None
    clip_text_path = paths.get_stage1_output_clip_text()
    if clip_text_path.exists():
        logger.info(f"加载CLIP文本提示: {clip_text_path}")
        clip_text_prompts = torch.load(clip_text_path, map_location='cpu')
    else:
        logger.warning(f"⚠️ CLIP文本提示文件不存在: {clip_text_path}")

    # 加载预处理数据（必需：阶段1和阶段3的输出）
    all_data_path = paths.get_stage1_output_all_data()
    all_predictions_path = paths.get_stage3_output_predictions()

    if not all_data_path.exists():
        raise FileNotFoundError(f"预处理数据文件不存在: {all_data_path}，请先运行阶段1")
    if not all_predictions_path.exists():
        raise FileNotFoundError(f"预测数据文件不存在: {all_predictions_path}，请先运行阶段3")

    logger.info("加载预处理数据")
    all_data = torch.load(all_data_path, map_location='cpu')
    all_predictions = torch.load(all_predictions_path, map_location='cpu')
    logger.info(f"✅ 加载完成: {len(all_data)} 张图像的数据，{len(all_predictions)} 张图像的预测")

    # 获取配置参数
    num_masks = config.get('training.stage4.num_masks', 5)
    min_scores = 0.05
    max_points = 1000

    # 开始推理
    results = {}
    image_list = [fname for fname in test_data.keys() if fname in all_data and fname in all_predictions]

    if len(image_list) == 0:
        logger.warning("⚠️ 测试数据中没有找到预处理过的图像，请检查图像文件名是否匹配")
        return results

    logger.info(f"开始处理 {len(image_list)} 张图像")

    for fname in tqdm.tqdm(image_list, desc="推理中"):
        try:
            # 1. 提取当前图像的特征并转移到GPU
            features = all_data[fname]['features'].cuda()

            # 2. 获取小样本提示或零样本文本
            with torch.no_grad():
                cls_head.eval()
                if mode == 'zeroshot':
                    # 零样本：使用CLIP文本提示
                    if clip_text_prompts is None:
                        logger.warning(f"图像 {fname} 零样本模式但无CLIP文本提示，跳过")
                        continue
                    class_name = all_data[fname].get('class_name', '')
                    if class_name not in clip_text_prompts:
                        logger.warning(f"图像 {fname} 类别 {class_name} 不在CLIP文本提示中，跳过")
                        continue
                    example_features = clip_text_prompts[class_name].unsqueeze(0).cuda()
                else:
                    # 小样本：使用示例框的CLIP特征
                    if 'example_clip_features' not in all_data[fname]:
                        logger.warning(f"图像 {fname} 小样本模式但无示例CLIP特征，跳过")
                        continue
                    example_features = all_data[fname]['example_clip_features'].cuda()

            # 3. 预测分数和框的筛选
            # 注意：这里应该使用all_predictions，不是all_data
            if 'pred_points_score' not in all_predictions[fname]:
                logger.warning(f"图像 {fname} 无pred_points_score，跳过")
                continue

            pred_points_score = all_predictions[fname]['pred_points_score']
            mask = torch.zeros(pred_points_score.size(0))
            mask[:min(pred_points_score.size(0), max_points)] = 1
            mask[pred_points_score < min_scores] = 0

            # 4. 根据掩码筛选预测框和IoU分数，并转移到GPU
            if 'pred_boxes' not in all_predictions[fname] or 'pred_ious' not in all_predictions[fname]:
                logger.warning(f"图像 {fname} 无pred_boxes或pred_ious，跳过")
                continue

            pred_boxes = all_predictions[fname]['pred_boxes'][:, :num_masks][mask.bool()].cuda()
            pred_ious = all_predictions[fname]['pred_ious'][:, :num_masks][mask.bool()].cuda()

            if len(pred_boxes) == 0:
                logger.warning(f"图像 {fname} 筛选后无候选框，跳过")
                results[fname] = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'count': 0
                }
                continue

            # 5. 使用ROIHead进行分类
            all_pred_boxes = []
            all_pred_scores = []

            for indices in torch.arange(len(pred_boxes)).split(128):
                with torch.no_grad():
                    # ROIHead推理（参考4_1_train_roi_head.py第251行）
                    # ROIHeadMLP同时支持few-shot和zero-shot模式
                    # 两种模式都使用相同的调用方式，区别在于example_features的来源：
                    # - few-shot模式：example_features来自示例框的CLIP特征 (all_data[fname]['example_clip_features'])
                    # - zero-shot模式：example_features来自CLIP文本提示 (clip_text_prompts[class_name])
                    cls_outs_ = cls_head(features, [pred_boxes[indices], ], [example_features, ] * len(indices))
                    # ROIHeadMLP返回的是logits，需要sigmoid后对每个mask的得分求平均（参考4_1_train_roi_head.py第252行）
                    pred_logits = cls_outs_.sigmoid().view(-1, len(example_features), num_masks).mean(1)

                    # 将预测得分与IoU得分相乘，作为最终得分
                    pred_logits = pred_logits * pred_ious[indices]

                    all_pred_boxes.append(pred_boxes[indices, torch.argmax(pred_logits, dim=1)])
                    all_pred_scores.append(pred_logits.max(dim=1).values)

            # 6. 后处理：缩放坐标到原图尺寸
            height, width = all_data[fname]['height'], all_data[fname]['width']
            scale = max(height, width) / 1024.
            pred_boxes = torch.cat(all_pred_boxes) * scale
            pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]].clamp(0, width)
            pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]].clamp(0, height)
            pred_scores = torch.cat(all_pred_scores)

            # 7. 过滤：面积过滤
            box_area = vision_ops.box_area(pred_boxes)
            mask = (box_area < (height * width * 0.75)) & (box_area > 10)
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]

            # 8. NMS
            if len(pred_boxes) > 0:
                nms_indices = vision_ops.nms(pred_boxes, pred_scores, 0.5)
                pred_boxes = pred_boxes[nms_indices]
                pred_scores = pred_scores[nms_indices]

            # 9. 保存预测结果
            image_id = int(fname[:-4]) if fname[:-4].isdigit() else hash(fname) % 1000000

            # from myutil.localization import evaluate_detection_metrics, get_pred_points_from_density
            # density_map = pred_scores.cpu().squeeze(0).squeeze(0)
            # gt_points = test_data[fname].get('points', [])
            # # ========== 关键修改3：使用缩放后的centers计算指标 ==========
            # f1, precision, recall = evaluate_detection_metrics(
            #     pred_density_map=density_map,
            #     gt_points=gt_points,  # 替换为缩放后的中心点
            #     distance_thresh=10.0
            # )

            #
            results[fname] = {
                'boxes': pred_boxes.cpu().numpy(),
                'scores': pred_scores.cpu().numpy(),
                'count': len(pred_boxes),
                'image_id': image_id
            }

        except Exception as e:
            logger.error(f"处理图像 {fname} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[fname] = {
                'boxes': np.array([]),
                'scores': np.array([]),
                'count': 0
            }

    logger.info(f"✅ 推理完成，处理了 {len(results)} 张图像")
    return results


def evaluate_results(
        results: Dict,
        test_data: Dict,
        dataset_type: str,
        output_dir: Path
):
    """
    评估推理结果

    Args:
        results: 推理结果字典
        test_data: 测试数据字典
        dataset_type: 数据集类型
        output_dir: 输出目录

    Returns:
        dict: 评估指标
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始评估")
    logger.info("=" * 60)

    # 计算计数指标
    mae_list = []
    rmse_list = []
    nae_list = []
    sre_list = []

    individual_results = []

    for fname, result in results.items():
        if fname not in test_data:
            continue

        # 真实计数
        gt_points = test_data[fname].get('points', [])
        gt_count = len(gt_points) if isinstance(gt_points, list) else 0

        # 预测计数
        pred_count = result.get('count', 0)

        # 计算误差
        err = abs(gt_count - pred_count)
        mae_list.append(err)
        rmse_list.append(err ** 2)

        if gt_count > 0:
            nae_list.append(err / gt_count)
            sre_list.append((err ** 2) / gt_count)

        individual_results.append({
            'filename': fname,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'error': err
        })

    # 计算总体指标
    metrics = {
        'MAE': np.mean(mae_list) if mae_list else 0.0,
        'RMSE': np.sqrt(np.mean(rmse_list)) if rmse_list else 0.0,
        'NAE': np.mean(nae_list) if nae_list else 0.0,
        'SRE': np.sqrt(np.mean(sre_list)) if sre_list else 0.0,
        'total_images': len(individual_results)
    }

    # 保存结果
    results_file = output_dir / 'inference_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'individual_results': individual_results
        }, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("评估结果:")
    logger.info("=" * 60)
    logger.info(f"  MAE (平均绝对误差): {metrics['MAE']:.4f}")
    logger.info(f"  RMSE (均方根误差): {metrics['RMSE']:.4f}")
    logger.info(f"  NAE (归一化平均误差): {metrics['NAE']:.4f}")
    logger.info(f"  SRE (平方相对误差): {metrics['SRE']:.4f}")
    logger.info(f"  总图像数: {metrics['total_images']}")
    logger.info("=" * 60)
    logger.info(f"结果已保存到: {results_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='PseCo推理测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件
  python test_pipeline.py --config config/default_config.yaml --test_data_root /path/to/test --anno_path /path/to/annotations.json

  # 指定模型权重
  python test_pipeline.py --test_data_root /path/to/test --anno_path /path/to/annotations.json --ckpt_path /path/to/checkpoint.pth

  # 指定数据集类型
  python test_pipeline.py --test_data_root /path/to/test --anno_path /path/to/annotations.json --dataset_type custom_json
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='配置文件路径（默认: config/default_config.yaml）'
    )
    parser.add_argument(
        '--test_data_root',
        type=str,
        required=True,
        help='测试数据根目录（图像所在目录）'
    )
    parser.add_argument(
        '--anno_path',
        type=str,
        required=True,
        help='标注文件路径'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='模型权重路径（如果为None，则从配置中获取）'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='auto',
        choices=['auto', 'coco', 'fsc147_json', 'custom_json'],
        help='数据集类型（默认: auto自动检测）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='输出目录（默认: results）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备（默认: cuda:0）'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='zeroshot',
        choices=['fewshot', 'zeroshot'],
        help='推理模式：fewshot（小样本）或zeroshot（零样本），默认: zeroshot'
    )

    args = parser.parse_args()

    # 设置日志
    logger, log_file = setup_logging()
    logger.info("=" * 60)
    logger.info("PseCo推理测试启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"测试数据根目录: {args.test_data_root}")
    logger.info(f"标注文件: {args.anno_path}")
    logger.info(f"数据集类型: {args.dataset_type}")
    logger.info(f"输出目录: {args.output_dir}")

    # 加载配置
    try:
        config = load_config(args.config)
        logger.info("✅ 配置加载成功")
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        return 1

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载测试数据
        test_data, detected_type = load_test_data(
            args.test_data_root,
            args.anno_path,
            args.dataset_type
        )

        # 加载模型权重
        checkpoints = load_model_checkpoints(config, args.ckpt_path)

        # 运行推理
        results = run_inference(
            config,
            test_data,
            args.test_data_root,
            checkpoints,
            args.device,
            args.mode
        )

        # 评估结果
        metrics = evaluate_results(
            results,
            test_data,
            detected_type,
            output_dir
        )

        logger.info("=" * 60)
        logger.info("✅ 推理测试完成")
        logger.info(f"  日志文件: {log_file}")
        logger.info(f"  结果目录: {output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"❌ 推理测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
