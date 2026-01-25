#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PseCo统一训练入口
支持一键运行4个训练阶段，支持断点续训和数据集配置
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.stage_checker import StageChecker
from utils.path_manager import PathManager

# 导入各阶段函数
from stages.stage1_generate_data import run_stage1
from stages.stage2_train_heatmap import run_stage2
from stages.stage3_extract_proposals import run_stage3
from stages.stage4_train_roi_head import run_stage4


def print_stage_status(checker: StageChecker):
    """打印各阶段完成状态"""
    status = checker.get_stage_status()
    stage_names = {
        '1': '数据预处理',
        '2': '训练点解码器',
        '3': '提取候选框',
        '4': '训练ROI分类头',
    }
    print("\n" + "=" * 60)
    print("训练阶段完成状态:")
    print("=" * 60)
    for stage_id, name in stage_names.items():
        completed = "✅" if status[stage_id] else "❌"
        print(f"  阶段{stage_id} ({name}): {completed}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='PseCo统一训练入口：一键运行4个训练阶段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 一键运行所有阶段
  python train_pipeline.py --config config/default_config.yaml

  # 只运行特定阶段
  python train_pipeline.py --stages 3 4

  # 断点续训（自动跳过已完成的阶段）
  python train_pipeline.py --resume

  # 使用新数据集配置
  python train_pipeline.py --config config/fscd_lvis_config.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='配置文件路径（默认: config/default_config.yaml）'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        default=['1', '2', '3', '4'],
        help='要执行的阶段列表（默认: 1 2 3 4）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点续训：自动跳过已完成的阶段'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查阶段完成状态，不执行训练'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新执行所有阶段（即使已完成）'
    )

    args = parser.parse_args()

    # 加载配置
    print(f"📁 加载配置文件: {args.config}")
    try:
        config = load_config(args.config)
        print(f"✅ 配置加载成功")
        print(f"   项目根目录: {config.get('project_root')}")
        print(f"   数据集: {config.get('dataset.name')}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return 1

    # 初始化检查器
    checker = StageChecker(config)

    # 打印阶段状态
    print_stage_status(checker)

    # 如果只是检查，则退出
    if args.check_only:
        return 0

    # 定义阶段映射
    stage_funcs = {
        '1': (run_stage1, '数据预处理'),
        '2': (run_stage2, '训练点解码器'),
        '3': (run_stage3, '提取候选框'),
        '4': (run_stage4, '训练ROI分类头'),
    }

    # 执行阶段
    success_count = 0
    skip_count = 0
    fail_count = 0

    for stage_id in args.stages:
        if stage_id not in stage_funcs:
            print(f"❌ 未知阶段: {stage_id}")
            fail_count += 1
            continue

        func, name = stage_funcs[stage_id]

        # 检查是否已完成
        is_completed = checker.check_stage(stage_id)
        if args.resume and is_completed and not args.force:
            print(f"⏭️  阶段{stage_id} ({name}) 已完成，跳过")
            skip_count += 1
            continue

        if args.force and is_completed:
            print(f"⚠️  阶段{stage_id} ({name}) 已完成，但使用--force强制重新执行")

        # 执行阶段
        print("\n" + "=" * 60)
        print(f"🚀 开始执行阶段{stage_id}: {name}")
        print("=" * 60)

        try:
            func(config)
            print(f"✅ 阶段{stage_id} ({name}) 完成")
            success_count += 1
        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断，阶段{stage_id} ({name}) 未完成")
            print("💡 提示: 使用 --resume 可以断点续训")
            return 1
        except Exception as e:
            print(f"❌ 阶段{stage_id} ({name}) 失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            if not args.resume:
                print("❌ 训练失败，退出")
                return 1
            else:
                print("⚠️  继续执行下一阶段...")

    # 打印总结
    print("\n" + "=" * 60)
    print("训练总结:")
    print("=" * 60)
    print(f"  ✅ 成功: {success_count} 个阶段")
    print(f"  ⏭️  跳过: {skip_count} 个阶段")
    print(f"  ❌ 失败: {fail_count} 个阶段")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())