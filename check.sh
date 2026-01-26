#!/bin/bash
echo "===== Detectron2 环境检查 ====="

# 1. 检查Python版本
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "1. Python 版本: $python_version"
if [[ $(echo "$python_version >= 3.6" | bc -l) -eq 1 ]]; then
  echo "   ✅ 满足（≥3.6）"
else
  echo "   ❌ 不满足（需要≥3.6）"
fi

# 2. 检查PyTorch版本
torch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo -e "\n2. PyTorch 版本: $torch_version"
if [[ $(echo "$torch_version >= 1.8" | bc -l) -eq 1 ]]; then
  echo "   ✅ 满足（≥1.8）"
else
  echo "   ❌ 不满足（需要≥1.8）"
fi

# 3. 检查torchvision是否安装
python -c "import torchvision" > /dev/null 2>&1
if [[ $? -eq 0 ]]; then
  echo -e "\n3. torchvision: ✅ 已安装"
else
  echo -e "\n3. torchvision: ❌ 未安装"
fi

# 4. 检查g++版本
gxx_version=$(g++ --version | grep -oP '(\d+\.\d+\.\d+)' | head -1 | cut -d. -f1-2)
echo -e "\n4. g++ 版本: $gxx_version"
if [[ $(echo "$gxx_version >= 5.4" | bc -l) -eq 1 ]]; then
  echo "   ✅ 满足（≥5.4）"
else
  echo "   ❌ 不满足（需要≥5.4）"
fi

# 5. 检查CUDA是否可用（可选但重要）
python -c "import torch; print('\n5. CUDA 可用:', torch.cuda.is_available())"

# 6. 检查ninja（推荐）
ninja --version > /dev/null 2>&1
if [[ $? -eq 0 ]]; then
  echo -e "\n6. ninja: ✅ 已安装"
else
  echo -e "\n6. ninja: ⚠️ 未安装（推荐安装：sudo apt install ninja-build）"
fi

echo -e "\n===== 检查完成 ====="