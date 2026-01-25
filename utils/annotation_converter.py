"""
标注格式转换工具
支持将自定义格式（JSON/TXT/XML）转换为COCO格式，用于评估
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnnotationConverter:
    """标注格式转换器：将各种格式转换为COCO格式"""
    
    @staticmethod
    def detect_format(annotation_path: str) -> str:
        """
        自动检测标注文件格式
        
        Args:
            annotation_path: 标注文件路径
            
        Returns:
            str: 格式类型 ('coco', 'fsc147_json', 'custom_json', 'txt', 'xml')
        """
        path = Path(annotation_path)
        
        if not path.exists():
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
        
        # 根据扩展名和内容判断
        if path.suffix == '.json':
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否为COCO格式
            if 'images' in data and 'annotations' in data and 'categories' in data:
                return 'coco'
            # 检查是否为FSC147格式
            elif isinstance(data, dict) and any('annotations' in v for v in data.values()):
                return 'fsc147_json'
            else:
                return 'custom_json'
        elif path.suffix == '.txt':
            return 'txt'
        elif path.suffix == '.xml':
            return 'xml'
        else:
            raise ValueError(f"不支持的标注格式: {path.suffix}")
    
    @staticmethod
    def convert_fsc147_to_coco(
        fsc147_json_path: str,
        output_path: str,
        image_dir: str,
        coco_gt_path: Optional[str] = None
    ) -> str:
        """
        将FSC147格式JSON转换为COCO格式（用于评估）
        
        FSC147格式：
        {
            "image1.jpg": {
                "width": 1024,
                "height": 768,
                "annotations": {
                    "0": {"points": [x, y]},
                    "1": {"points": [x, y]}
                },
                "box_examples_coordinates": [[x1,y1,x2,y2], ...],
                "class_name": "car",
                "split": "train"
            }
        }
        
        Args:
            fsc147_json_path: FSC147格式JSON文件路径
            output_path: 输出COCO格式文件路径
            image_dir: 图像目录
            coco_gt_path: 如果已有COCO格式GT文件，直接使用（可选）
            
        Returns:
            str: 输出文件路径
        """
        # 如果已有COCO格式GT文件，直接返回
        if coco_gt_path and Path(coco_gt_path).exists():
            logger.info(f"使用已有的COCO格式GT文件: {coco_gt_path}")
            return coco_gt_path
        
        logger.info(f"开始转换FSC147格式到COCO格式: {fsc147_json_path}")
        
        # 加载FSC147格式数据
        with open(fsc147_json_path, 'r', encoding='utf-8') as f:
            fsc147_data = json.load(f)
        
        # 构建COCO格式
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object", "supercategory": "none"}]
        }
        
        image_id_map = {}  # 文件名 -> image_id映射
        ann_id = 1
        
        for idx, (fname, target) in enumerate(fsc147_data.items()):
            # 添加图像信息
            image_id = int(fname.replace('.jpg', '').replace('.png', '')) if fname[:-4].isdigit() else idx + 1
            image_id_map[fname] = image_id
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": fname,
                "width": target.get('width', 0),
                "height": target.get('height', 0)
            })
            
            # 添加标注信息（基于points生成边界框）
            # 注意：FSC147只有点标注，需要转换为框标注
            # 这里使用点坐标作为中心，生成小框（8x8像素）
            annotations = target.get('annotations', {})
            if isinstance(annotations, dict):
                for ann_key, ann_value in annotations.items():
                    if 'points' in ann_value:
                        point = ann_value['points']
                        if isinstance(point, list) and len(point) == 2:
                            x, y = point[0], point[1]
                            # 生成小框（8x8像素）
                            bbox = [float(x - 4), float(y - 4), 8.0, 8.0]
                            
                            coco_data["annotations"].append({
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": bbox,  # [x, y, w, h]
                                "area": 64.0,
                                "iscrowd": 0
                            })
                            ann_id += 1
        
        # 保存COCO格式文件
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ COCO格式文件已保存: {output_path}")
        logger.info(f"   图像数量: {len(coco_data['images'])}")
        logger.info(f"   标注数量: {len(coco_data['annotations'])}")
        
        return output_path
    
    @staticmethod
    def convert_custom_json_to_coco(
        custom_json_path: str,
        output_path: str,
        image_dir: str,
        point_to_bbox_size: int = 8
    ) -> str:
        """
        将自定义JSON格式转换为COCO格式
        
        自定义格式示例：
        {
            "image1.jpg": {
                "width": 1024,
                "height": 768,
                "points": [[x1, y1], [x2, y2], ...],
                "boxes": [[x1, y1, x2, y2], ...]  # 可选
            }
        }
        
        Args:
            custom_json_path: 自定义JSON文件路径
            output_path: 输出COCO格式文件路径
            image_dir: 图像目录
            point_to_bbox_size: 点标注转换为框标注时的框大小（像素）
        """
        logger.info(f"开始转换自定义JSON格式到COCO格式: {custom_json_path}")
        
        with open(custom_json_path, 'r', encoding='utf-8') as f:
            custom_data = json.load(f)
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object", "supercategory": "none"}]
        }
        
        ann_id = 1
        
        for idx, (fname, target) in enumerate(custom_data.items()):
            image_id = idx + 1
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": fname,
                "width": target.get('width', 0),
                "height": target.get('height', 0)
            })
            
            # 优先使用boxes，如果没有则从points生成
            if 'boxes' in target and target['boxes']:
                boxes = target['boxes']
                for box in boxes:
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        area = (x2 - x1) * (y2 - y1)
                        
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": float(area),
                            "iscrowd": 0
                        })
                        ann_id += 1
            elif 'points' in target and target['points']:
                # 从点生成小框
                points = target['points']
                for point in points:
                    if len(point) == 2:
                        x, y = point[0], point[1]
                        bbox = [float(x - point_to_bbox_size/2), 
                               float(y - point_to_bbox_size/2),
                               float(point_to_bbox_size),
                               float(point_to_bbox_size)]
                        
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": float(point_to_bbox_size ** 2),
                            "iscrowd": 0
                        })
                        ann_id += 1
        
        # 保存
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ COCO格式文件已保存: {output_path}")
        return output_path
    
    @staticmethod
    def convert_to_coco(
        annotation_path: str,
        output_path: str,
        image_dir: str,
        format_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        通用转换接口：自动检测格式并转换
        
        Args:
            annotation_path: 输入标注文件路径
            output_path: 输出COCO格式文件路径
            image_dir: 图像目录
            format_type: 指定格式类型（可选，auto为自动检测）
            **kwargs: 其他参数
            
        Returns:
            str: 输出文件路径
        """
        if format_type is None or format_type == 'auto':
            format_type = AnnotationConverter.detect_format(annotation_path)
        
        logger.info(f"检测到标注格式: {format_type}")
        
        if format_type == 'coco':
            # 已经是COCO格式，直接复制
            import shutil
            shutil.copy(annotation_path, output_path)
            logger.info(f"✅ COCO格式文件已复制: {output_path}")
            return output_path
        elif format_type == 'fsc147_json':
            return AnnotationConverter.convert_fsc147_to_coco(
                annotation_path, output_path, image_dir, 
                kwargs.get('coco_gt_path')
            )
        elif format_type == 'custom_json':
            return AnnotationConverter.convert_custom_json_to_coco(
                annotation_path, output_path, image_dir,
                kwargs.get('point_to_bbox_size', 8)
            )
        else:
            raise ValueError(f"暂不支持该格式的转换: {format_type}")


# 便捷函数
def convert_annotations_to_coco(
    annotation_path: str,
    output_path: str,
    image_dir: str,
    format_type: str = 'auto'
) -> str:
    """
    便捷函数：转换标注为COCO格式
    
    Args:
        annotation_path: 输入标注文件路径
        output_path: 输出COCO格式文件路径
        image_dir: 图像目录
        format_type: 格式类型（auto/coco/fsc147_json/custom_json）
        
    Returns:
        str: 输出文件路径
    """
    return AnnotationConverter.convert_to_coco(
        annotation_path, output_path, image_dir, format_type
    )
