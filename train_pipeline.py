#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PseCo统一训练入口
支持一键运行4个训练阶段，支持断点续训和数据集配置

功能：
1. 支持一次性执行所有4个训练阶段，也可通过命令行参数指定单独阶段（--stage 1/2/3/4/all）
2. 断点续训：每个阶段训练前自动检测断点文件（ckpt/stageX_ckpt.pth），有则加载断点继续，无则从头开始
3. 报错日志：添加完整的异常捕获和日志输出，记录每个阶段的训练步数、损失值、报错行号/函数名
"""
import argparse
import sys
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime

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


def setup_logging(config, log_file: str = None):
    """
    设置日志系统
    
    Args:
        config: 配置对象
        log_file: 日志文件路径（如果为None，则使用配置中的路径）
    """
    if log_file is None:
        log_dir = Path(config.get('outputs.log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志处理器：同时输出到文件和终端
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger, log_file


def print_stage_status(checker: StageChecker, logger: logging.Logger):
    """打印各阶段完成状态"""
    status = checker.get_stage_status()
    stage_names = {
        '1': '数据预处理',
        '2': '训练点解码器',
        '3': '提取候选框',
        '4': '训练ROI分类头',
    }
    
    logger.info("=" * 60)
    logger.info("训练阶段完成状态:")
    logger.info("=" * 60)
    for stage_id, name in stage_names.items():
        completed = "✅" if status[stage_id] else "❌"
        logger.info(f"  阶段{stage_id} ({name}): {completed}")
    logger.info("=" * 60)


def check_checkpoint(config, stage_id: str) -> str:
    """
    检查阶段断点文件
    
    Args:
        config: 配置对象
        stage_id: 阶段ID ('1', '2', '3', '4')
        
    Returns:
        str: 断点文件路径，如果不存在则返回None
    """
    paths = PathManager(config)
    checkpoint_dir = paths.get_checkpoint_dir()
    
    # 不同阶段的断点文件命名
    checkpoint_files = {
        '1': None,  # 阶段1无训练，无断点
        '2': checkpoint_dir / 'point_decoder_vith_v5.pth',  # 阶段2断点
        '3': None,  # 阶段3是分布式推理，断点逻辑在脚本内部
        '4': None,  # 阶段4断点逻辑在脚本内部（每1000次迭代保存）
    }
    
    checkpoint_path = checkpoint_files.get(stage_id)
    if checkpoint_path and checkpoint_path.exists():
        return str(checkpoint_path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='PseCo统一训练入口：一键运行4个训练阶段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 一键运行所有阶段
  python train_pipeline.py --config config/default_config.yaml --stage all

  # 只运行特定阶段
  python train_pipeline.py --stage 1
  python train_pipeline.py --stage 2
  python train_pipeline.py --stage 3
  python train_pipeline.py --stage 4

  # 运行多个阶段
  python train_pipeline.py --stage 3 4

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
        '--stage',
        type=str,
        nargs='+',
        default=['all'],
        help='要执行的阶段（1/2/3/4/all），可指定多个，如: --stage 1 2 或 --stage all'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点续训：自动跳过已完成的阶段，并加载断点文件继续训练'
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
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='指定日志文件路径（默认: logs/train_log_TIMESTAMP.txt）'
    )

    args = parser.parse_args()
    
    # 解析阶段参数
    if 'all' in args.stage:
        stages_to_run = ['1', '2', '3', '4']
    else:
        stages_to_run = [s for s in args.stage if s in ['1', '2', '3', '4']]
        if not stages_to_run:
            print("❌ 错误: 无效的阶段参数，必须是 1/2/3/4/all")
            return 1

    # 加载配置
    try:
        config = load_config(args.config)
        print(f"📁 加载配置文件: {args.config}")
        print(f"✅ 配置加载成功")
        print(f"   项目根目录: {config.get('project_root')}")
        print(f"   数据集: {config.get('dataset.name')}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 设置日志
    logger, log_file = setup_logging(config, args.log_file)
    logger.info("=" * 60)
    logger.info("PseCo训练流程启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"执行阶段: {stages_to_run}")
    logger.info(f"断点续训: {args.resume}")
    logger.info(f"强制重新执行: {args.force}")

    # 初始化检查器
    checker = StageChecker(config)

    # 打印阶段状态
    print_stage_status(checker, logger)

    # 如果只是检查，则退出
    if args.check_only:
        logger.info("仅检查模式，退出")
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

    for stage_id in stages_to_run:
        if stage_id not in stage_funcs:
            logger.error(f"未知阶段: {stage_id}")
            fail_count += 1
            continue

        func, name = stage_funcs[stage_id]

        # 检查是否已完成
        is_completed = checker.check_stage(stage_id)
        if args.resume and is_completed and not args.force:
            logger.info(f"⏭️  阶段{stage_id} ({name}) 已完成，跳过")
            skip_count += 1
            continue

        if args.force and is_completed:
            logger.warning(f"⚠️  阶段{stage_id} ({name}) 已完成，但使用--force强制重新执行")

        # 检查断点文件
        checkpoint_path = None
        if args.resume:
            checkpoint_path = check_checkpoint(config, stage_id)
            if checkpoint_path:
                logger.info(f"📦 检测到断点文件: {checkpoint_path}")
                logger.info(f"   将从此断点继续训练阶段{stage_id}")
            else:
                logger.info(f"📦 未检测到断点文件，将从开头开始训练阶段{stage_id}")

        # 执行阶段
        logger.info("=" * 60)
        logger.info(f"🚀 开始执行阶段{stage_id}: {name}")
        logger.info("=" * 60)
        
        stage_start_time = datetime.now()
        
        try:
            # 传递断点路径给阶段函数（如果支持）
            if checkpoint_path and stage_id in ['2', '4']:
                func(config, checkpoint_path=checkpoint_path)
            else:
                func(config)
            
            stage_end_time = datetime.now()
            duration = (stage_end_time - stage_start_time).total_seconds()
            
            logger.info(f"✅ 阶段{stage_id} ({name}) 完成")
            logger.info(f"   耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
            success_count += 1
            
        except KeyboardInterrupt:
            logger.warning(f"\n⚠️  用户中断，阶段{stage_id} ({name}) 未完成")
            logger.info("💡 提示: 使用 --resume 可以断点续训")
            return 1
            
        except Exception as e:
            stage_end_time = datetime.now()
            duration = (stage_end_time - stage_start_time).total_seconds()
            
            # 获取详细的错误信息
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            
            logger.error("=" * 60)
            logger.error(f"❌ 阶段{stage_id} ({name}) 失败")
            logger.error("=" * 60)
            logger.error(f"错误类型: {exc_type.__name__}")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"耗时: {duration:.2f} 秒")
            logger.error("=" * 60)
            logger.error("详细错误堆栈:")
            logger.error("=" * 60)
            for line in tb_lines:
                logger.error(line.rstrip())
            logger.error("=" * 60)
            
            # 提取关键错误位置信息
            if exc_traceback:
                tb = exc_traceback
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                logger.error(f"错误位置: {frame.f_code.co_filename}:{tb.tb_lineno}")
                logger.error(f"错误函数: {frame.f_code.co_name}")
            
            fail_count += 1
            if not args.resume:
                logger.error("❌ 训练失败，退出")
                return 1
            else:
                logger.warning("⚠️  继续执行下一阶段...")

    # 打印总结
    total_time = (datetime.now() - stage_start_time).total_seconds() if 'stage_start_time' in locals() else 0
    
    logger.info("=" * 60)
    logger.info("训练总结:")
    logger.info("=" * 60)
    logger.info(f"  ✅ 成功: {success_count} 个阶段")
    logger.info(f"  ⏭️  跳过: {skip_count} 个阶段")
    logger.info(f"  ❌ 失败: {fail_count} 个阶段")
    logger.info(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())