#!/bin/bash

# ViT Test 프로젝트 가상환경 활성화 스크립트

echo "🐍 ViT Test 프로젝트 가상환경 활성화"
echo "=========================================="

# 가상환경 활성화
source .venv/bin/activate

echo "✅ 가상환경이 활성화되었습니다!"
echo ""
echo "📦 설치된 주요 패키지:"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  - NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""
echo "🚀 사용 가능한 명령어:"
echo "  python train_vit_hf.py        # 기본 ViT 훈련"
echo "  python run_vit.py --help      # 실행 옵션 확인"
echo "  python setup_tokens.py       # 토큰 설정"
echo ""
echo "💡 가상환경 비활성화: deactivate"
echo ""

# 새로운 셸 시작 (가상환경이 활성화된 상태로)
exec $SHELL
