import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Rank %(rank)d] - %(message)s',
    handlers=[
        logging.FileHandler('extract_proposals.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 创建自定义logger类以支持rank信息
class DistributedLogger:
    def __init__(self, rank):
        self.rank = rank

    def info(self, message):
        logging.info(message, extra={'rank': self.rank})

    def warning(self, message):
        logging.warning(message, extra={'rank': self.rank})

    def error(self, message):
        logging.error(message, extra={'rank': self.rank})

sys.path.insert(0, '/mnt/mydisk/wjj/PseCo-main/')
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

# 初始化logger
rank = dist.get_rank()
world_size = dist.get_world_size()
logger = DistributedLogger(rank)

logger.info(f"初始化分布式环境: Rank={rank}, World Size={world_size}")


def chunks(arr, m):
    import math
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


# 模型加载
logger.info("开始加载模型...")
from models import PointDecoder

project_root = '/mnt/mydisk/wjj/PseCo-main/'

logger.info("构建SAM模型...")
sam = build_sam_vit_h("/mnt/mydisk/wjj/Prompt_sam_localization/checkpoint/sam_vit_h_4b8939.pth").cuda().eval()
logger.info("构建PointDecoder模型...")
point_decoder = PointDecoder(sam).cuda().eval()
logger.info("加载PointDecoder权重...")
state_dict = torch.load(f'{project_root}/data/fsc147/checkpoints/point_decoder_vith.pth',
                        map_location='cpu')
point_decoder.load_state_dict(state_dict)
logger.info("模型加载完成")

# 数据加载
logger.info("开始加载数据...")
all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith_v5.pth', map_location='cpu')
logger.info(f"数据加载完成，总共有 {len(all_data)} 张图像")

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
logger.info("开始为PointDecoder准备输入数据...")
# 1. 为每张图像生成256x256的掩码（Mask）
for fname in tqdm.tqdm(all_data, desc=f"Rank {rank}: Preparing masks"):
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
logger.info("输入数据准备完成")

# 2. 划分数据集（train/val/test/all）
logger.info("开始划分数据集...")
all_image_list = {'train': [], 'val': [], 'test': [], 'all': []}
for fname in all_data:
    # if all_data[fname]['split'] == 'train':
    #     if (all_data[fname]['annotations']['points'].size(0) + all_data[fname]['segment_anything']['points'].size(
    #             0)) != 0:
    #         all_image_list[all_data[fname]['split']].append(fname)
    # else:
    all_image_list[all_data[fname]['split']].append(fname)
    all_image_list['all'].append(fname)
logger.info(f"数据集划分完成: train={len(all_image_list['train'])}, val={len(all_image_list['val'])}, test={len(all_image_list['test'])}, all={len(all_image_list['all'])}")

# 3. 为当前进程分配图像列表（数据分片）
logger.info(f"为当前进程(Rank {rank})分配数据...")
all_file_names = chunks(all_image_list['all'], dist.get_world_size())[dist.get_rank()]
logger.info(f"当前进程分配到 {len(all_file_names)} 张图像")

# 4. 初始化推理结果字典（如果已有部分结果，直接加载避免重复计算）
save_path = f'{base_path}_{dist.get_rank()}.pth'

if os.path.exists(save_path):
    logger.info(f"检测到已存在的结果文件: {save_path}，正在加载...")
    predictions = torch.load(save_path, map_location='cpu')
    logger.info(f"已加载 {len(predictions)} 个已处理的图像结果")
else:
    predictions = {}
    logger.info("未找到已存在的结果文件，将从头开始处理")

logger.info(f"当前进程将处理 {len(all_file_names)} 张图像，其中 {len([f for f in all_file_names if f not in predictions])} 张待处理")

# ===================== 核心推理流程（PointDecoder+SAM 生成目标点和框） =====================

logger.info(f"Rank {rank}: 开始核心推理流程...")
print(dist.get_rank(), len(all_file_names))

# 计算待处理图像数量
remaining_files = [fname for fname in all_file_names if fname not in predictions]
logger.info(f"Rank {rank}: 待处理图像数量: {len(remaining_files)}")

for n_iter, fname in enumerate(tqdm.tqdm(all_file_names, desc=f"Rank {rank}: Processing")):
    if fname in predictions:
        continue

    logger.info(f"Rank {rank}: 处理图像 {fname} ({n_iter+1}/{len(remaining_files)})")

    try:
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

        logger.info(f"Rank {rank}: 从图像 {fname} 检测到 {len(pred_points)} 个候选点")

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

        logger.info(f"Rank {rank}: 图像 {fname} 处理完成，生成 {len(pred_boxes)} 个预测框")
        # ========== 新增：显存释放逻辑 ==========
        del features, outputs_heatmaps, pred_points, pred_points_score  # 删除变量
        torch.cuda.empty_cache()  # 清空GPU缓存
        import gc
        gc.collect()  # 强制垃圾回收
        # ================ 每100张图像保存一次，防止意外中断==================
        if n_iter % 100 == 0:
            logger.info(f"Rank {rank}: 保存中间结果，已处理 {n_iter+1} 张图像")
            torch.save(predictions, save_path)
    except Exception as e:
        logger.error(f"Rank {rank}: 处理图像 {fname} 时发生错误: {str(e)}")
        continue

logger.info(f"Rank {rank}: 核心推理流程完成，已处理 {len(predictions)} 张图像")

# ===================== CLIP 特征提取 + 分布式结果合并（最终输出） =====================

logger.info(f"Rank {rank}: 开始CLIP特征提取...")
from ops.foundation_models import clip
import numpy as np

clip.available_models()

logger.info(f"Rank {rank}: 加载CLIP模型...")
model, preprocess = clip.load("/mnt/mydisk/wjj/online_models/torch_cache/hub/checkpoints/ViT-B-32.pt")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

logger.info(f"Rank {rank}: CLIP模型参数统计:")
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
import torchvision

normalize = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711))


def read_image(fname):
    img = Image.open(f'/mnt/mydisk/wjj/dataset/FSC_147/images_384_VarV2/{fname}')
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
        for indices in torch.arange(len(examples)).split(64):
            e.append(model.encode_image(examples[indices].cuda()).float())
    e = torch.cat(e, dim=0)
    e = F.normalize(e, dim=1).cpu()
    return e

# 入口：为每个图像的候选框提取CLIP特征
logger.info(f"Rank {rank}: 开始为 {len(all_file_names)} 张图像提取CLIP特征...")
processed_count = 0
for n_iter, fname in enumerate(tqdm.tqdm(all_file_names, desc=f"Rank {rank}: CLIP extraction")):
    if 'clip_regions' in predictions[fname]:
        logger.info(f"Rank {rank}: 图像 {fname} 的CLIP特征已存在，跳过")
        continue

    logger.info(f"Rank {rank}: 为图像 {fname} 提取CLIP特征")
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

    logger.info(f"Rank {rank}: 图像 {fname} 的CLIP特征提取完成，提取了 {len(boxes)} 个候选框的特征")

    processed_count += 1
    if n_iter % 100 == 0:
        logger.info(f"Rank {rank}: 保存CLIP特征提取中间结果，已处理 {processed_count} 张图像")
        torch.save(predictions, save_path)

logger.info(f"Rank {rank}: CLIP特征提取完成")

# 保存最终结果
logger.info(f"Rank {rank}: 保存最终结果到 {save_path}")
torch.save(predictions, save_path)
logger.info(f"Rank {rank}: 最终结果已保存")

# ===================== 合并所有GPU进程的结果 =====================

# 等待所有进程完成
logger.info(f"Rank {rank}: 等待所有进程完成...")
dist.barrier()

if rank == 0:  # 只在主进程合并结果
    logger.info("开始合并所有GPU进程的结果...")
    predictions = {}
    total_processed = 0

    for i in range(world_size):
        logger.info(f"加载 Rank {i} 的结果...")
        data = torch.load(f'{base_path}_{i}.pth', map_location='cpu')
        for fname in data:
            predictions[fname] = data[fname]
        logger.info(f"已加载 Rank {i} 的 {len(data)} 个结果")
        total_processed += len(data)

    logger.info(f"合并完成，总共处理了 {total_processed} 张图像")
    logger.info(f"保存合并后的最终结果到 {base_path + '.pth'}")
    torch.save(predictions, base_path + '.pth')
    logger.info("所有处理完成！")
else:
    logger.info(f"Rank {rank}: 等待主进程完成合并...")

logger.info(f"Rank {rank}: 进程完成")
