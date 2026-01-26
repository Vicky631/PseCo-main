"""
检查各训练阶段的完成状态
"""
import torch
from pathlib import Path
from utils.path_manager import PathManager
from utils.config_loader import ConfigLoader


class StageChecker:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.paths = PathManager(config)

    def check_stage1(self) -> bool:
        """检查阶段1是否完成"""
        required_files = [
            self.paths.get_stage1_output_all_data(),
            self.paths.get_stage1_output_segment_data(),
            self.paths.get_stage1_output_pseudo_boxes(),
            self.paths.get_stage1_output_clip_text(),
        ]
        return all(f.exists() and f.stat().st_size > 0 for f in required_files)

    def check_stage2(self) -> bool:
        """检查阶段2是否完成"""
        checkpoint = self.paths.get_stage2_output_point_decoder()
        if not checkpoint.exists():
            return False
        # 检查checkpoint是否包含模型权重
        try:
            state_dict = torch.load(checkpoint, map_location='cpu')
            if isinstance(state_dict, dict):
                # 可能是 {'point_decoder': ..., 'optimizer': ...} 格式
                return 'point_decoder' in state_dict or any('transformer' in k for k in state_dict.keys())
            return True
        except:
            return False

    def check_stage3(self) -> bool:
        """检查阶段3是否完成"""
        predictions_file = self.paths.get_stage3_output_predictions()
        if not predictions_file.exists():
            return False
        # 检查文件是否有效
        try:
            data = torch.load(predictions_file, map_location='cpu')
            return isinstance(data, dict) and len(data) > 0
        except:
            return False

    def check_stage4(self) -> bool:
        """检查阶段4是否完成（检查是否有最终checkpoint）"""
        run_name = self.config.get('training.stage4.run_name')
        if self.config.get('training.stage4.mode') == 'zeroshot':
            run_name += '_zeroshot'
        output_dir = self.paths.get_stage4_output_dir(run_name)
        # 检查是否有checkpoint文件
        checkpoint_files = list(output_dir.glob('*.pth')) if output_dir.exists() else []
        return len(checkpoint_files) > 0

    def check_stage(self, stage_id: str) -> bool:
        """检查指定阶段是否完成"""
        checkers = {
            '1': self.check_stage1,
            '2': self.check_stage2,
            '3': self.check_stage3,
            '4': self.check_stage4,
        }
        if stage_id not in checkers:
            raise ValueError(f"未知阶段: {stage_id}")
        return checkers[stage_id]()

    def get_stage_status(self) -> dict:
        """获取所有阶段的完成状态"""
        return {
            '1': self.check_stage1(),
            '2': self.check_stage2(),
            '3': self.check_stage3(),
            '4': self.check_stage4(),
        }