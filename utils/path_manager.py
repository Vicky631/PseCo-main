"""
统一路径管理器：基于配置生成所有路径
"""
from utils.config_loader import ConfigLoader
from pathlib import Path
from typing import Optional


class PathManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.project_root = Path(config.get('project_root'))
        self.dataset_root = Path(config.get('dataset.root'))
        self.dataset_name = config.get('dataset.name')

    # ========== 数据集路径 ==========
    def get_annotation_file(self) -> Path:
        return Path(self.config.get('dataset.annotation_file'))

    def get_image_dir(self) -> Path:
        return Path(self.config.get('dataset.image_dir'))

    def get_coco_annotation(self) -> Path:
        return Path(self.config.get('dataset.coco_annotation'))

    # ========== 模型路径 ==========
    def get_sam_checkpoint(self) -> Path:
        return Path(self.config.get('models.sam.checkpoint'))

    def get_clip_checkpoint(self) -> Path:
        return Path(self.config.get('models.clip.checkpoint'))

    # ========== 输出路径 ==========
    def get_data_dir(self) -> Path:
        data_dir = Path(self.config.get('outputs.data_dir'))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_checkpoint_dir(self) -> Path:
        checkpoint_dir = Path(self.config.get('outputs.checkpoint_dir'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def get_log_dir(self) -> Path:
        log_dir = Path(self.config.get('outputs.log_dir'))
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    # ========== 阶段输出文件 ==========
    def get_stage1_output_all_data(self, version: str = 'vith_v5_fix') -> Path:
        """阶段1输出：all_data文件"""
        return self.get_data_dir() / f'all_data_{version}.pth'

    def get_stage1_output_segment_data(self) -> Path:
        """阶段1输出：segment_anything_data"""
        return self.get_data_dir() / 'segment_anything_data_vith.pth'

    def get_stage1_output_pseudo_boxes(self) -> Path:
        """阶段1输出：pseudo_boxes"""
        return self.get_data_dir() / 'pseudo_boxes_data_vith.pth'

    def get_stage1_output_clip_text(self) -> Path:
        """阶段1输出：CLIP文本提示"""
        return self.get_data_dir().parent / 'clip_text_prompt.pth'

    def get_stage2_output_point_decoder(self, version: str = 'vith_v5') -> Path:
        """阶段2输出：point_decoder权重"""
        checkpoint_dir = self.get_checkpoint_dir()
        return checkpoint_dir / f'point_decoder_{version}.pth'

    def get_stage3_output_predictions(self) -> Path:
        """阶段3输出：all_predictions"""
        return self.get_data_dir() / 'all_predictions_vith.pth'

    def get_stage4_output_dir(self, run_name: str) -> Path:
        """阶段4输出：ROI分类头权重目录"""
        return self.get_checkpoint_dir() / 'cls_head' / 'ckpt' / run_name