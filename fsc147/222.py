# import os
# import sys
# import torch
# import logging
#
# # 配置日志（方便看结果）
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# # 项目根路径（和你的训练代码保持一致）
# project_root = '/mnt/mydisk/wjj/PseCo-main'
# sys.path.insert(0, project_root)
#
#
# # ========== 核心：加载数据并扫描异常 ==========
# def check_example_clip_features():
#     # 1. 加载all_data（和训练代码用同一个文件）
#     logger.info("开始加载all_data文件...")
#     all_data = torch.load(f'{project_root}/data/fsc147/sam/all_data_vith_v5_fix.pth', map_location='cpu')
#     logger.info(f"all_data加载完成，共{len(all_data)}个文件")
#
#     # 2. 初始化异常记录
#     abnormal_records = []
#     total_files = len(all_data)
#     checked_files = 0
#
#     # 3. 遍历所有文件，检查example_clip_features
#     for fname, data in all_data.items():
#         checked_files += 1
#         if checked_files % 100 == 0:
#             logger.info(f"已检查{checked_files}/{total_files}个文件，当前异常数：{len(abnormal_records)}")
#
#         # 跳过没有example_clip_features的文件
#         if 'example_clip_features' not in data:
#             logger.warning(f"文件{fname}无example_clip_features，跳过")
#             continue
#
#         # 获取example_clip_features
#         clip_feats = data['example_clip_features']
#         # 确保是tensor（避免其他格式）
#         if not isinstance(clip_feats, torch.Tensor):
#             logger.error(f"文件{fname}的example_clip_features不是Tensor，类型：{type(clip_feats)}")
#             continue
#
#         # 打印该文件的特征基本信息（前100个文件打印，避免日志过多）
#         if checked_files <= 100:
#             logger.debug(f"文件{fname} - example_clip_features形状：{clip_feats.shape}")
#
#         # 4. 检查每个特征的维度（重点：第0维是否为4，期望是3）
#         # clip_feats的形状应该是 [N, 3, 512]（N是特征数量，3是关键维度，512是特征长度）
#         if len(clip_feats.shape) != 3:
#             # 维度数异常
#             abnormal_records.append({
#                 "file": fname,
#                 "error_type": "维度数异常",
#                 "total_shape": clip_feats.shape,
#                 "expected_dim": 3,
#                 "actual_dim": len(clip_feats.shape)
#             })
#             logger.error(
#                 f"❌ 文件{fname} - example_clip_features维度数异常：期望3维，实际{len(clip_feats.shape)}维，形状{clip_feats.shape}")
#             continue
#
#         # 检查第1维（关键维度）是否为4（你的问题是第1维=4，期望=3）
#         feat_num, key_dim, feat_len = clip_feats.shape
#         if key_dim != 3:
#             # 关键维度异常（重点排查）
#             abnormal_records.append({
#                 "file": fname,
#                 "error_type": "关键维度异常",
#                 "total_shape": clip_feats.shape,
#                 "expected_key_dim": 3,
#                 "actual_key_dim": key_dim,
#                 "feat_len": feat_len
#             })
#             logger.error(
#                 f"❌ 文件{fname} - example_clip_features关键维度异常：期望第1维=3，实际={key_dim}，形状{clip_feats.shape}")
#
#             # 进一步检查：是否有个别特征维度异常（比如大部分3维，个别4维）
#             # 遍历每个特征（可选，耗时但精准）
#             for feat_idx in range(feat_num):
#                 single_feat = clip_feats[feat_idx]
#                 if len(single_feat.shape) != 2 or single_feat.shape[0] != 3:
#                     logger.error(f"   → 特征{feat_idx}形状异常：{single_feat.shape}，期望(3,512)")
#
#     # 5. 输出最终汇总
#     logger.info("\n===== 排查结果汇总 =====")
#     if abnormal_records:
#         logger.error(f"共发现{len(abnormal_records)}个文件的example_clip_features异常：")
#         for rec in abnormal_records:
#             logger.error(f"文件：{rec['file']} | 问题：{rec['error_type']} | 详情：{rec}")
#     else:
#         logger.info("✅ 所有文件的example_clip_features维度均正常（第1维=3）")
#
#
# # ========== 运行排查 ==========
# if __name__ == "__main__":
#     check_example_clip_features()


import sys
import torch
# 打印当前Python解释器路径（必须包含WJJldm）
print("当前Python路径：", sys.executable)
# 验证是否能导入albumentations
try:
    import albumentations as A
    print("✅ albumentations导入成功，版本：", A.__version__)
except ImportError as e:
    print("❌ 导入失败：", e)
    # 打印当前解释器对应的pip安装包（确认是否有albumentations）
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "show", "albumentations"])