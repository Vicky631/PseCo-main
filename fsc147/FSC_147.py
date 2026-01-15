import os
import json
from typing import Optional, Union, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


class FSC147DatasetLoader:
    """
    FSC147数据集加载类
    功能：加载FSC147的标注信息和图像，返回原图尺寸/缩放后尺寸的points和boxes，支持tensor/numpy格式转换
    """

    def __init__(self,
                 annotation_file: str,
                 image_root: str,
                 max_size: int = 1024):
        """
        初始化加载器
        Args:
            annotation_file: FSC147标注文件路径（json格式）
            image_root: 图像根目录
            max_size: 缩放的最大尺寸（原图会按 max_size/max(w,h) 缩放，保持比例）
        """
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.max_size = max_size

        # 加载标注文件
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # 预处理所有标注（缓存缩放后的信息，避免重复计算）
        self._preprocess_annotations()

    def _preprocess_annotations(self):
        """预处理所有标注，计算缩放比例并缓存"""
        self.annotation_cache = {}
        for fname in tqdm(self.annotations.keys(), desc="Preprocessing FSC147 annotations"):
            target = self.annotations[fname]
            # 计算缩放比例（原图 -> max_size尺寸）
            orig_w, orig_h = target['width'], target['height']
            scale = self.max_size / max(orig_w, orig_h)

            # 提取并缩放points
            points_list = [target['annotations'][l]['points'] for l in target['annotations']]
            points_tensor = torch.tensor(points_list, dtype=torch.float32)

            # 提取并缩放boxes（box_examples_coordinates）
            boxes_tensor = torch.tensor(target['box_examples_coordinates'], dtype=torch.float32)

            # 缓存信息：原图尺寸、缩放比例、原图points/boxes、缩放后points/boxes
            self.annotation_cache[fname] = {
                'orig_size': (orig_w, orig_h),  # 原图尺寸 (w, h)
                'scale': scale,  # 缩放比例
                'orig_points': points_tensor,  # 原图尺寸的points (N, 2)
                'orig_boxes': boxes_tensor,  # 原图尺寸的boxes (M, 4) [x1,y1,x2,y2]
                'scaled_points': points_tensor * scale,  # 缩放后的points
                'scaled_boxes': boxes_tensor * scale  # 缩放后的boxes
            }

    def get_image(self, fname: str, return_scaled: bool = False) -> Union[Image.Image, Tuple[Image.Image, float]]:
        """
        加载图像
        Args:
            fname: 图像文件名（如 '2.jpg'）
            return_scaled: 是否返回缩放后的图像+缩放比例，否则返回原图
        Returns:
            原图 或 (缩放后的图像, 缩放比例)
        """
        img_path = os.path.join(self.image_root, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"加载图像 {fname} 失败: {e}")

        if not return_scaled:
            return img

        # 缩放图像（保持比例，最大边为max_size）
        orig_w, orig_h = img.size
        scale = self.max_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

        # 补全到max_size×max_size（可选，根据你的需求）
        img_padded = Image.new("RGB", (self.max_size, self.max_size), (255, 255, 255))
        img_padded.paste(img_scaled, (0, 0))

        return img_padded, scale

    def get_annotations(self,
                        fname: str,
                        return_scaled: bool = False,
                        return_numpy: bool = False) -> dict:
        """
        获取指定图像的标注信息
        Args:
            fname: 图像文件名（如 '2.jpg'）
            return_scaled: 是否返回缩放后的标注（否则返回原图尺寸）
            return_numpy: 是否返回numpy数组（否则返回torch.Tensor）
        Returns:
            dict: 包含以下键
                - orig_size: 原图尺寸 (w, h)
                - scale: 缩放比例
                - points: 点标注 (N, 2)
                - boxes: 框标注 (M, 4) [x1,y1,x2,y2]
        """
        if fname not in self.annotation_cache:
            raise KeyError(f"未找到 {fname} 的标注信息")

        cache = self.annotation_cache[fname]
        result = {
            'orig_size': cache['orig_size'],
            'scale': cache['scale']
        }

        # 选择返回原图/缩放后的标注
        if return_scaled:
            points = cache['scaled_points']
            boxes = cache['scaled_boxes']
        else:
            points = cache['orig_points']
            boxes = cache['orig_boxes']

        # 转换为numpy（如果需要）
        if return_numpy:
            points = points.numpy()
            boxes = boxes.numpy()

        result['points'] = points
        result['boxes'] = boxes

        return result

    def get_all_filenames(self) -> list:
        """获取所有图像文件名列表"""
        return list(self.annotations.keys())

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """按索引获取数据（兼容PyTorch Dataset）"""
        fname = self.get_all_filenames()[idx]
        img = self.get_image(fname)
        anns = self.get_annotations(fname)

        return {
            'filename': fname,
            'image': img,
            'orig_size': anns['orig_size'],
            'scale': anns['scale'],
            'points': anns['points'],
            'boxes': anns['boxes']
        }


# -------------------------- 测试示例 --------------------------
def visualize_annotations(img, points, boxes, save_path=None):
    """
    在图像上可视化标注点和框（无窗口版，仅保存图像）
    Args:
        img: PIL.Image 对象（原图）
        points: 点标注 (N,2) tensor/numpy，格式 [x, y]
        boxes: 框标注 (M,4) tensor/numpy，格式 [x1, y1, x2, y2]
        save_path: 保存路径（必须指定，否则无输出）
    """
    if save_path is None:
        print("⚠️ 无桌面环境下必须指定 save_path，跳过可视化")
        return

    # 转换为 OpenCV 格式（BGR）
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 转换为 numpy 数组（如果输入是 tensor）
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()

    # 绘制标注点（红色圆点，半径5）
    for (x, y) in points:
        cv2.circle(img_cv, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    # 绘制框（绿色矩形，线宽2）
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)

    # 仅保存图像，不显示窗口
    cv2.imwrite(save_path, img_cv)
    print(f"✅ 可视化图像已保存至: {save_path}")


# -------------------------- 主测试逻辑 --------------------------
if __name__ == "__main__":
    # 初始化加载器
    loader = FSC147DatasetLoader(
        annotation_file='/home/zy/wjj/PseCo-main/data/fsc147/annotation_FSC147_384_with_gt.json',
        image_root='/home/zy/wjj/PseCo-main/data/fsc147/images_384_VarV2',
        max_size=1024
    )

    # 单张图像测试：指定测试文件名
    test_fname = '7049.jpg'

    # 1. 加载原图和原图标注（tensor格式）
    img_orig = loader.get_image(test_fname)
    anns_orig = loader.get_annotations(test_fname, return_scaled=False, return_numpy=False)

    # 打印原图标注信息
    print(f"\n=== {test_fname} 原图标注信息 ===")
    print(f"原图尺寸 (w, h): {anns_orig['orig_size']}")
    print(f"缩放比例 (max_size/原图最大边): {anns_orig['scale']:.4f}")
    print(f"Points 形状 (tensor, 原图尺寸): {anns_orig['points'].shape}")
    print(f"Boxes 形状 (tensor, 原图尺寸): {anns_orig['boxes'].shape}")

    # 2. 可视化并保存原图标注结果
    visualize_annotations(
        img=img_orig,
        points=anns_orig['points'],
        boxes=anns_orig['boxes'],
        save_path=f'./visualization_{test_fname}'  # 保存到当前目录
    )

    # 3. 加载缩放后图像和标注（tensor格式）
    img_scaled, scale = loader.get_image(test_fname, return_scaled=True)
    anns_scaled = loader.get_annotations(test_fname, return_scaled=True, return_numpy=False)

    # 打印缩放后标注信息
    print(f"\n=== {test_fname} 缩放后标注信息 ===")
    print(f"缩放后图像尺寸: {img_scaled.size}")
    print(f"Points 形状 (tensor, 缩放后): {anns_scaled['points'].shape}")
    print(f"Boxes 形状 (tensor, 缩放后): {anns_scaled['boxes'].shape}")

    # 4. 可视化并保存缩放后标注结果
    visualize_annotations(
        img=img_scaled,
        points=anns_scaled['points'],
        boxes=anns_scaled['boxes'],
        save_path=f'./visualization_{test_fname}_scaled.jpg'
    )

    # 5. 可选：按索引测试（兼容PyTorch DataLoader）
    # 找到test_fname对应的索引
    all_fnames = loader.get_all_filenames()
    if test_fname in all_fnames:
        test_idx = all_fnames.index(test_fname)
        sample = loader[test_idx]
        print(f"\n=== 索引{test_idx}的样本信息（对应{test_fname}） ===")
        print(f"文件名: {sample['filename']}")
        print(f"Points 形状: {sample['points'].shape}")
        print(f"Boxes 形状: {sample['boxes'].shape}")
    else:
        print(f"\n⚠️ {test_fname} 不在数据集列表中，索引测试跳过")    # print(f"Boxes shape: {sample['boxes'].shape}")