# 1. 导入detectron2核心模块
try:
    import detectron2
    print(f"✅ detectron2基础导入成功，版本：{detectron2.__version__}")
except ImportError as e:
    print(f"❌ detectron2导入失败：{e}")

# 2. 导入依赖CUDA的核心模块（这是之前报错的关键）
try:
    from detectron2.layers import DeformConv, ModulatedDeformConv
    from detectron2.structures import BoxMode
    print("✅ detectron2 CUDA相关模块导入成功")
except ImportError as e:
    print(f"❌ CUDA模块导入失败：{e}")