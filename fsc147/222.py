# import os
# import shutil
# import random
# from pathlib import Path
#
# # 指定文件路径
# image_path = '/home/zy/wjj/PseCo-main/data/fsc147/images_384_VarV2'
# label_path = '/home/zy/wjj/PseCo-main/data/fsc147/label'
# gt_path = '/home/zy/wjj/PseCo-main/data/fsc147/gt'
#
# # 获取所有文件名（不带后缀）
# image_files = [f.stem for f in Path(image_path).glob('*.*')]
# label_files = [f.stem for f in Path(label_path).glob('*.*')]
# gt_files = [f.stem for f in Path(gt_path).glob('*.*')]
#
# # 确保图片、标签和gt文件名相同
# assert image_files == label_files == gt_files, "文件名不匹配！"
#
# # 打乱文件顺序
# random.shuffle(image_files)
#
# # 划分比例
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15
#
# # 计算划分索引
# total_files = len(image_files)
# train_end = int(total_files * train_ratio)
# val_end = int(total_files * (train_ratio + val_ratio))
#
# # 划分数据
# train_files = image_files[:train_end]
# val_files = image_files[train_end:val_end]
# test_files = image_files[val_end:]
#
# # 创建目标文件夹
# for split in ['/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/Train', '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/Val', '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/Test']:
#     os.makedirs(f'{split}/images', exist_ok=True)
#     os.makedirs(f'{split}/labels', exist_ok=True)
#     os.makedirs(f'{split}/gt', exist_ok=True)
#
# # 将文件复制到对应文件夹
# def copy_files(files, split):
#     for file in files:
#         shutil.copy(os.path.join(image_path, f"{file}.jpg"), f'{split}/images/')
#         shutil.copy(os.path.join(label_path, f"{file}.txt"), f'{split}/labels/')
#         shutil.copy(os.path.join(gt_path, f"{file}.png"), f'{split}/gt/')
#
# copy_files(train_files, 'train')
# copy_files(val_files, 'val')
# copy_files(test_files, 'test')
#
# print(f"划分完成：{len(train_files)} 训练集, {len(val_files)} 验证集, {len(test_files)} 测试集")
#

import os
import json
import shutil
from importlib.util import source_hash

from torch.fx.experimental.unification.multipledispatch.dispatcher import source

# 定义路径


# import os
# import shutil
#
# # 定义路径
# source_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/train/images'  # 替换为源文件夹路径
# target_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/train/ground_truth'  # 替换为目标文件夹路径
# destination_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/train/ground_truth/1'  # 替换为存放多余文件的目标文件夹路径
#
# # 创建目标文件夹（如果不存在）
# os.makedirs(destination_dir, exist_ok=True)
#
# # 获取源文件夹和目标文件夹中的文件名（不带扩展名）
# source_files = {os.path.splitext(f)[0] for f in os.listdir(source_dir)}
# target_files = {os.path.splitext(f)[0] for f in os.listdir(target_dir)}
#
# # 比较两个文件夹，找到多余的文件
# extra_files_in_source = source_files - target_files
# extra_files_in_target = target_files - source_files
#
# # 转移多余的文件到目标文件夹
# def move_extra_files(extra_files, from_dir):
#     for file_name in extra_files:
#         # 在源文件夹或目标文件夹中查找文件（考虑扩展名）
#         for ext in ['.jpg', '.png', '.jpeg']:  # 根据实际情况添加文件扩展名
#             file_path = os.path.join(from_dir, file_name + ext)
#             if os.path.exists(file_path):
#                 # 构建目标文件路径并转移文件
#                 destination_file = os.path.join(destination_dir, file_name + ext)
#                 shutil.move(file_path, destination_file)
#                 print(f'文件 {file_path} 已转移到 {destination_file}')
#                 break
#
# # 转移多余的文件
# move_extra_files(extra_files_in_source, source_dir)
# move_extra_files(extra_files_in_target, target_dir)
#
# print('文件转移完成')

test_list=[]
train_list=[]
val_list=[]
files1 = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/train/ground_truth'
train_names = [os.path.splitext(file)[0] for file in os.listdir(files1) if os.path.isfile(os.path.join(files1, file))]
files2 = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/test/ground_truth'
test_names = [os.path.splitext(file)[0] for file in os.listdir(files2) if os.path.isfile(os.path.join(files2, file))]
files3 = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC147/valid/ground_truth'
val_names = [os.path.splitext(file)[0] for file in os.listdir(files3) if os.path.isfile(os.path.join(files3, file))]

source_file='/home/zy/wjj/PseCo-main/data/fsc147/MT'

# 目标文件夹路径
train_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC/train/ground_truth'
test_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC/test/ground_truth'
val_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/FSC/valid/ground_truth'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 将文件按照名称复制到对应的文件夹
for file in os.listdir(source_file):
    source_path = os.path.join(source_file, file)

    # 检查是否是文件
    if os.path.isfile(source_path):

        file_name_without_extension = os.path.splitext(file)[0]
        file_name_without_extension=file_name_without_extension.replace('_density','').replace('_center_points','')
        # 依据文件名决定目标文件夹
        if file_name_without_extension in train_names:
            destination = os.path.join(train_dir, file)
        elif file_name_without_extension in test_names:
            destination = os.path.join(test_dir, file)
        elif file_name_without_extension in val_names:
            destination = os.path.join(val_dir, file)
        else:
            continue  # 如果文件名不在任何一个列表中，则跳过

        # 复制文件到目标路径
        shutil.copy(source_path, destination.replace('_density',''))

print("文件复制完成！")