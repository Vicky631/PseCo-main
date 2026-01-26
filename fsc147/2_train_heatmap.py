"""
文件功能：基于Segment Anything Model (SAM) + 自定义点解码器，完成FSC147数据集的点预测与热力图训练
核心目标：针对FSC147计数数据集，利用SAM提取的特征训练点解码器，实现目标关键点的预测并生成热力图
适用场景：小样本视觉计数任务中，基于SAM的关键点检测与热力图生成

==================== 输入输出说明 ====================
【输入】
1. 数据集文件：
   - /mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/all_data_vith_v5.pth：FSC147数据集的基础信息（图像名、标注点、分割结果、特征等）
   - /mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/segment_anything_data_vith.pth：SAM模型对FSC147图像的分割结果（预测框、预测点）
   - 图像文件路径：/mnt/mydisk/wjj/PseCo-main/data/fsc147/images_384_VarV2/ （FSC147原始图像）
2. 模型相关：
   - SAM-ViT-H模型（也可切换ViT-L/ViT-B）：用于提取图像特征
   - 自定义PointDecoder模型：用于关键点预测的解码器

【输出】
1. 训练产物：
   - /mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/checkpoints/point_decoder_vith_v5.pth：训练好的点解码器权重文件（包含模型参数、优化器状态）
2. 可视化结果：
   - 原始图像叠加SAM预测框/点的可视化图
   - 关键点热力图（真实热力图+模型预测热力图）
   - 模型预测关键点的可视化图
3. 中间产物：
   - 每个图像的256x256掩码（target['mask']）
   - 关键点热力图（256x256）
   - 模型预测的关键点坐标（pred_points）、预测热力图（pred_heatmaps）
"""
import os
import sys

sys.path.insert(0, '/mnt/mydisk/wjj/PseCo-main')
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
# 开启CUDA内核缓存（指定缓存目录）
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch.backends.cudnn.benchmark = True  # 开启cudnn基准测试，加速CUDA内核选择
torch.backends.cudnn.enabled = True
# 设置CUDA内核缓存目录（可选，默认在~/.cache/torch）
import os
os.environ['TORCH_EXTENSIONS_DIR'] = '/mnt/mydisk/wjj/torch_cache/'  # 自定义缓存目录，避免权限问题
os.makedirs('/mnt/mydisk/wjj/torch_cache/', exist_ok=True)
from PIL import Image
import numpy as np
import tqdm
import albumentations as A
from ops.ops import plot_results

plt.rcParams["figure.dpi"] = 300

torch.cuda.set_device(0)
torch.autograd.set_grad_enabled(False)

from ops.foundation_models.segment_anything import build_sam_vit_h
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sam = build_sam_vit_h("/mnt/mydisk/wjj/Prompt_sam_localization/checkpoint/sam_vit_h_4b8939.pth").cuda().eval()


# sam = build_sam_vit_l().cuda().eval()
# sam = build_sam_vit_b().cuda().eval()

def read_image(fname):
    img = Image.open(f'/mnt/mydisk/wjj/dataset/FSC_147/images_384_VarV2/{fname}')
    transform = A.Compose([
        A.LongestMaxSize(1024),
        A.PadIfNeeded(1024, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    img = Image.fromarray(transform(image=np.array(img))['image'])
    return img


# # all_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/all_data_vitb.pth', map_location='cpu')
# all_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/all_data_vith_v5.pth', map_location='cpu')
# # all_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/all_data_vitl.pth', map_location='cpu')
# # segment_anthing_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/segment_anything_data_vith.pth',
#                                   map_location='cpu')
logger.info("开始加载数据...")
all_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/all_data_vith_v5.pth', map_location='cpu')
logger.info(f"数据加载完成，共{len(all_data)}个样本")
segment_anthing_data = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/segment_anything_data_vith.pth',
                                  map_location='cpu')
logger.info("segment anything数据加载完成")

logger.info("开始数据预处理...")
for fname in tqdm.tqdm(all_data):
    target = all_data[fname]
    target['image_id'] = fname
    if fname in segment_anthing_data:
        target['segment_anything'] = segment_anthing_data[fname]
    transform = A.Compose([
        A.LongestMaxSize(256),
        A.PadIfNeeded(256, 256, border_mode=0, position=A.PadIfNeeded.PositionType.TOP_LEFT),
    ])
    mask = Image.fromarray(
        transform(image=np.ones((target['height'], target['width'])).astype(np.uint8) * 255)['image'])
    mask = np.array(mask) > 128
    target['mask'] = torch.from_numpy(mask).reshape(1, 1, 256, 256).bool().float()
logger.info("数据预处理完成")

logger.info("开始数据集划分...")
all_image_list = {'train': [], 'val': [], 'test': []}
for fname in all_data:
    if all_data[fname]['split'] == 'train':
        if (all_data[fname]['annotations']['points'].size(0) + all_data[fname]['segment_anything']['pred_points'].size(
                0)) != 0:
            all_image_list[all_data[fname]['split']].append(fname)
    else:
        all_image_list[all_data[fname]['split']].append(fname)
logger.info(
    f"数据集划分完成: train={len(all_image_list['train'])}, val={len(all_image_list['val'])}, test={len(all_image_list['test'])}")

fname = all_image_list['train'][3]
plot_results(read_image(fname),
             #  bboxes=all_data[fname]['annotations']['boxes'],
             #  points=all_data[fname]['annotations']['points'],
             bboxes=all_data[fname]['segment_anything']['pred_boxes'],
             points=all_data[fname]['segment_anything']['pred_points'],
             )


def extract_heatmap(points, sigma=2):
    scale = 4
    # sigma = 2
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.ones(len(points)).cuda() * sigma
    points = points / scale
    points = points.long().float()
    # x = torch.arange(0, 256, 1).cuda() + 0.5
    # y = torch.arange(0, 256, 1).cuda() + 0.5
    x = torch.arange(0, 256, 1).cuda()
    y = torch.arange(0, 256, 1).cuda()
    x, y = torch.meshgrid(x, y, indexing='xy')
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    heatmaps = torch.zeros(1, 1, 256, 256).cuda()
    for indices in torch.arange(len(points)).split(256):
        mu_x, mu_y = points[indices, 0].view(-1, 1, 1), points[indices, 1].view(-1, 1, 1)
        heatmaps_ = torch.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[indices].view(-1, 1, 1) ** 2))
        heatmaps_ = torch.max(heatmaps_, dim=0).values
        heatmaps_ = heatmaps_.reshape(1, 1, 256, 256)
        heatmaps = torch.maximum(heatmaps, heatmaps_)
    return heatmaps.float()


def create_heatmap(fname):
    t = all_data[fname]
    points = torch.cat([t['annotations']['points'].cuda(), t['segment_anything']['pred_points'].cuda(), ])
    st = len(t['annotations']['points'].cuda())
    min_sigma = 2.
    sigma = torch.ones(len(points)).cuda() * min_sigma
    for i in range(len(t['segment_anything']['pred_points'])):
        from ops.ops import gaussian_radius
        import math
        bbox = t['segment_anything']['pred_boxes'][i] / 4
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        sigma_ = (2 * radius + 1) / 6.0
        sigma_ = max(sigma_, min_sigma)
        sigma[st + i] = sigma_
    heatmaps = extract_heatmap(points, sigma)
    return heatmaps


fname = all_image_list['train'][15]
heatmaps = create_heatmap(fname)
plt.imshow(read_image(fname).resize((256, 256)))
plt.imshow(heatmaps[:, 0].squeeze().cpu(), alpha=0.4)

# ===================== 训练点解码器 =====================

# 初始化模型和训练相关参数
from ops.loggerx import LoggerX
from ops.ops import convert_to_cuda

logger = LoggerX(save_root=None, print_freq=10)
print('1')
from models import PointDecoder
print('2')
point_mask_decoder = PointDecoder(sam).cuda()
print('3')
logger.modules = [point_mask_decoder]
optimizer = torch.optim.AdamW(list(point_mask_decoder.parameters()), lr=0.0001, weight_decay=0.0)
print('4')
max_iter = 50000
acc_grd_step = 1
print('5')
from ops.grad_scaler import NativeScalerWithGradNormCount
print('6')
amp = True
scaler = NativeScalerWithGradNormCount(amp=amp)
print('7')
# bs = 32
bs = 64

print(f"准备训练，最大迭代次数: {max_iter}, 批次大小: {bs}")
train_indices = torch.randint(0, len(all_image_list['train']), (max_iter * bs,)).split(bs)
n_iter = 0

# 开始训练
point_mask_decoder_model_save_path= '/mnt/mydisk/wjj/PseCo-main/data/fsc147/sam/checkpoints/point_decoder_vith_v5.pth'
# 提取文件夹路径
save_dir = os.path.dirname(point_mask_decoder_model_save_path)
# 自动创建文件夹（如果不存在）
os.makedirs(save_dir, exist_ok=True)

print("开始训练...")
for n_iter, _indices in enumerate(train_indices):

    targets = [all_data[all_image_list['train'][i]] for i in _indices]
    targets = convert_to_cuda(targets)

    point_mask_decoder.train()

    with torch.autograd.set_grad_enabled(True) and torch.autocast(device_type='cuda', enabled=amp):

        pred_heatmaps = point_mask_decoder(torch.cat([t['features'] for t in targets]))['pred_heatmaps']
        heatmaps = torch.cat([create_heatmap(t['image_id']) for t in targets])
        masks = torch.cat([t['mask'] for t in targets]).flatten(1)
        # loss = vision_ops.sigmoid_focal_loss(pred_heatmaps, heatmaps, reduction='mean')
        # loss = F.binary_cross_entropy_with_logits(pred_heatmaps, heatmaps, reduction='mean')
        loss = F.mse_loss(pred_heatmaps, heatmaps, reduction='none').flatten(1)
        loss = ((loss * masks).sum(1) / (1e-5 + masks.sum(1))).mean()

        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        scaler(loss, optimizer=optimizer, update_grad=update_params)
    # 保存模型
    if (n_iter + 1) % 1000 == 0:
        torch.save({'point_decoder': point_mask_decoder.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   point_mask_decoder_model_save_path,
                   pickle_protocol=5
                   )
    logger.msg([loss, ], n_iter)
print("训练完成")
# 加载上面训练的模型
from models import PointDecoder

point_mask_decoder = PointDecoder(sam).cuda().eval()
state_dict = torch.load('/mnt/mydisk/wjj/PseCo-main/data/fsc147/checkpoints/point_decoder_vith.pth', map_location='cpu')
# point_mask_decoder.load_state_dict(state_dict['point_decoder'])
point_mask_decoder.load_state_dict(state_dict)

# 进行验证并可视化
fname = all_image_list['val'][3]
with torch.no_grad():
    point_mask_decoder.max_points = 256
    point_mask_decoder.nms_kernel_size = 3
    point_mask_decoder.point_threshold = 0.05
    outputs = point_mask_decoder(all_data[fname]['features'].cuda())
print(outputs['pred_points'].size())
plt.imshow(read_image(fname).resize((256, 256)))
plt.imshow(outputs['pred_heatmaps_nms'].squeeze().cpu().numpy(), alpha=0.5)

plot_results(read_image(fname),
             points=outputs['pred_points'].squeeze(),
             )
