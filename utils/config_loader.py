"""
配置加载器：加载YAML配置并解析路径模板
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
        self.config = self._resolve_paths(self.config)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解析路径模板变量"""
        # 先解析project_root和dataset.root
        project_root = config.get('project_root', '')
        dataset_root = config.get('dataset', {}).get('root', '')
        dataset_name = config.get('dataset', {}).get('name', '')

        # 递归替换所有路径模板
        def replace_vars(obj, context=None):
            if context is None:
                context = {
                    'project_root': project_root,
                    'dataset.root': dataset_root,
                    'dataset.name': dataset_name,
                }

            if isinstance(obj, dict):
                return {k: replace_vars(v, context) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_vars(item, context) for item in obj]
            elif isinstance(obj, str):
                # 替换模板变量
                for key, value in context.items():
                    obj = obj.replace(f'{{{key}}}', str(value))
                return obj
            else:
                return obj

        return replace_vars(config)

    def get(self, key: str, default=None):
        """获取配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str):
        """支持字典式访问"""
        return self.get(key)


def load_config(config_path: str = 'config/default_config.yaml') -> ConfigLoader:
    """加载配置文件的便捷函数"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    return ConfigLoader(config_path)