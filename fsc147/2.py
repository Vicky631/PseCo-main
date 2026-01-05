import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
import scipy.io
from tqdm import tqdm

class FSC147ProcessorFor256:
    def __init__(self, annotation_file, image_root, save_density_dir, save_mat_dir, save_resized_dir,
                 sigma=5):
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.save_density_dir = save_density_dir
        self.save_mat_dir = save_mat_dir
        self.save_resized_dir = save_resized_dir
        self.sigma = sigma

        os.makedirs(save_density_dir, exist_ok=True)
        os.makedirs(save_mat_dir, exist_ok=True)
        os.makedirs(save_resized_dir, exist_ok=True)

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def generate_density_map(self, points, H, W):
        density = np.zeros((H, W), dtype=np.float32)
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                density[y, x] += 1
        return gaussian_filter(density, sigma=self.sigma)

    def process_single(self, fname):
        target = self.annotations[fname]
        scale = 1  # No scaling needed, use original size

        # points is already in a tensor
        points = torch.tensor(
            [target['annotations'][l]['points'] for l in target['annotations']],
            dtype=torch.float32
        ) * scale

        img_path = os.path.join(self.image_root, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return

        # Get original image size
        W,H = img.size

        # 保存原图
        resized_img_path = os.path.join(self.save_resized_dir, fname)
        img.save(resized_img_path)

        # 生成密度图（与原图大小一致）
        d1 = self.generate_density_map(points, H, W)
        full_density = d1

        # 生成密度图与原图尺寸一致
        density_img = (full_density / (full_density.max() + 1e-8) * 255).astype(np.uint8)

        # 保存 png 密度图
        density_png_path = os.path.join(self.save_density_dir, fname.replace('.jpg', '_density.png'))
        Image.fromarray(density_img).save(density_png_path)

        # 保存中心点信息到 mat
        centers_np = points.numpy()
        mat_data = {'image_info': np.array([[centers_np]], dtype=object)}
        mat_path = os.path.join(self.save_mat_dir, fname.replace('.jpg', '_center_points.mat'))
        scipy.io.savemat(mat_path, mat_data)

        # 生成叠加图：原图和密度图叠加
        # 将密度图转换为三通道（通过复制密度图到3个通道）
        density_img_colored = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)

        # 确保两个图像的尺寸一致
        resized_img_cv2 = np.array(img)  # 转换为NumPy数组
        resized_img_cv2 = cv2.resize(resized_img_cv2, (density_img_colored.shape[1], density_img_colored.shape[0]))

        # # 叠加图：将彩色密度图和原图进行加权叠加
        # overlay = cv2.addWeighted(resized_img_cv2, 0.7, density_img_colored, 0.3, 0)
        #
        # # 保存叠加图
        # overlay_img_path = os.path.join(self.save_resized_dir, fname.replace('.jpg', '_overlay.png'))
        # # cv2.imwrite(overlay_img_path, overlay)

        print(f"✅ {fname} done: density -> {density_png_path}, mat -> {mat_path}, resized -> {resized_img_path}")


processor = FSC147ProcessorFor256(
    annotation_file='/home/zy/wjj/PseCo-main/data/fsc147/annotation_FSC147_384_with_gt.json',
    image_root='/home/zy/wjj/PseCo-main/data/fsc147/images_384_VarV2',
    save_density_dir='/home/zy/wjj/PseCo-main/data/fsc147/GT',
    save_mat_dir='/home/zy/wjj/PseCo-main/data/fsc147/MT',
    save_resized_dir='/home/zy/wjj/PseCo-main/data/fsc147/IMG',
)

# # 单张测试
# processor.process_single('9.jpg')

# 批量处理
for fname in tqdm(processor.annotations):
    processor.process_single(fname)
