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
    def __init__(self, annotation_file, image_root, save_density_dir, save_mat_dir,
                 output_size=1024, small_size=256, sigma=15):
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.save_density_dir = save_density_dir
        self.save_mat_dir = save_mat_dir
        self.output_size = output_size
        self.small_size = small_size
        self.sigma = sigma

        os.makedirs(save_density_dir, exist_ok=True)
        os.makedirs(save_mat_dir, exist_ok=True)

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def pad_and_resize(self, img):
        w, h = img.size
        scale = self.output_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (self.output_size, self.output_size))
        new_img.paste(img_resized, (0, 0))
        return new_img, scale

    def generate_density_map(self, points, H, W):
        density = np.zeros((H, W), dtype=np.float32)
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                density[y, x] += 1
        return gaussian_filter(density, sigma=self.sigma)

    def process_single(self, fname):
        target = self.annotations[fname]
        scale = 1024 / max(target['width'], target['height'])
        target['annotations'] = {
            'points': torch.Tensor([target['annotations'][l]['points'] for l in target['annotations']]).float() * scale}
        target['box_examples_coordinates'] = torch.Tensor(target['box_examples_coordinates']).float() * scale

        img_path = os.path.join(self.image_root, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return

        img, scale = self.pad_and_resize(img)
        H, W = self.output_size, self.output_size

        resized_img = img.resize((1024, 1024), Image.BILINEAR)
        resized_img_path = os.path.join(self.save_resized_dir, fname.replace('.jpg', '_1024.jpg'))
        resized_img.save(resized_img_path)
        # 点信息
        points = torch.tensor(
            [target['annotations'][l]['points'] for l in target['annotations']],
            dtype=torch.float32
        ) * scale

        # box中心点信息
        # boxes = torch.tensor(target['box_examples_coordinates'], dtype=torch.float32) * scale
        # centers = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        # centers = torch.stack(centers, dim=1)

        # 生成密度图
        d1 = self.generate_density_map(points, H, W)
        # d2 = self.generate_density_map(centers, H, W)
        full_density = d1

        # 生成 256x256 缩放密度图
        small_density = T.Resize((self.small_size, self.small_size))(torch.from_numpy(full_density).unsqueeze(0)).squeeze(0).numpy()
        density_img = (small_density / (small_density.max() + 1e-8) * 255).astype(np.uint8)

        # 保存 png 密度图
        density_png_path = os.path.join(self.save_density_dir, fname.replace('.jpg', '.png'))
        Image.fromarray(density_img).save(density_png_path)

        # 保存中心点信息到 mat
        # all_points = torch.cat([points, centers], dim=0)
        centers_np = points.numpy()
        mat_data = {'image_info': np.array([[centers_np]], dtype=object)}
        mat_path = os.path.join(self.save_mat_dir, fname.replace('.jpg', '.mat'))
        scipy.io.savemat(mat_path, mat_data)

        print(f"✅ {fname} done: density -> {density_png_path}, mat -> {mat_path}")


processor = FSC147ProcessorFor256(
    annotation_file='/home/zy/wjj/PseCo-main/data/fsc147/annotation_FSC147_384_with_gt.json',
    image_root='/home/zy/wjj/PseCo-main/data/fsc147/images_384_VarV2',
    save_density_dir='/home/zy/wjj/PseCo-main/data/fsc147/density_png_256',
    save_mat_dir='/home/zy/wjj/PseCo-main/data/fsc147/mat_center_points',
)

# 单张测试
processor.process_single('2.jpg')

# 你之后也可以批量处理：
# for fname in tqdm(processor.annotations):
#     processor.process_single(fname)



# import scipy.io
# import numpy as np
# import cv2
#
# #
# def read_mat_and_generate_gt_plot(file_path, save_path):
#     data = scipy.io.loadmat(file_path)
#     if 'image_info' in data:
#         # 提取点坐标，考虑额外维度
#         points = data['image_info'][0][0]
#     else:
#         raise KeyError("未在 .mat 文件中找到 'image_info' 键。")
#
#     # 扁平化点坐标数组（每个点是一个 [x, y] 的列表）
#     points = points.reshape(-1, 2)
#
#     # 创建一个 1024x1024 的白色背景图像
#     large_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # 白色背景
#
#     # 将点绘制在 1024x1024 图像上
#     for point in points:
#         x, y = int(point[0]), int(point[1])
#         # 在图像上画红色小圆点
#         cv2.circle(large_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # 红色
#
#     # 缩放图像到 256x256
#     small_image = cv2.resize(large_image, (256, 256), interpolation=cv2.INTER_LINEAR)
#
#     # 保存缩放后的图像
#     cv2.imwrite(save_path, small_image)
#     print(f"散点图已保存至: {save_path}")
#     print(f"总点数: {len(points)}")
#
# # 示例用法（替换为你的实际 .mat 文件路径）
# mat_file_path = '/home/zy/wjj/PseCo-main/data/fsc147/mat_center_points/2.mat'  # 替换为实际路径
# save_path = '/home/zy/wjj/PseCo-main/data/fsc147/mat_center_points/2.png'  # 指定保存图像的路径
# read_mat_and_generate_gt_plot(mat_file_path, save_path)
