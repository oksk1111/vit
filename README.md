# CIFAR-10 새/개 분류 및 멀티모달 모델 성능 비교

이 프로젝트는 CIFAR-10 데이터셋에서 새(bird)와 개(dog) 이미지를 분류하는 다양한 모델 아키텍처의 성능을 비교합니다. Vision Transformer (ViT), CNN, 텍스트 특성과의 멀티모달 모델 등 여러 접근 방식을 구현하고 그 성능을 평가합니다.

## 🚀 주요 기능

- **멀티모달 모델 비교**: ViT, CNN, 텍스트 기반, 그리고 다양한 융합 모델 비교
- **체크포인트 자동 관리**: 훈련 중단 후 재개 및 손상된 체크포인트 복구 기능
- **테스트 모드 지원**: 빠른 개발을 위한 경량 학습 모드
- **시각적 평가**: 혼동 행렬 및 예측 결과 시각화
- **견고한 에러 처리**: 학습 과정에서 발생하는 오류 자동 복구

## 📋 구현된 모델

1. **텍스트 전용 모델 (Text Only)**: 이미지의 텍스트 특성만을 사용하는 모델
2. **CNN 이미지 모델 (CNN Image)**: 표준 CNN 기반 이미지 분류기
3. **ViT 전용 모델 (ViT Only)**: Hugging Face의 Vision Transformer 기반 모델
4. **조기 융합 모델 (Early Fusion)**: 초기 단계에서 이미지와 텍스트 특성을 결합
5. **후기 융합 모델 (Late Fusion)**: 각 모델의 특성을 추출 후 최종 단계에서 결합
6. **어텐션 융합 모델 (Attention Fusion)**: 어텐션 메커니즘을 통해 동적으로 특성 융합

## 📦 설치

```bash
# 저장소 클론
git clone https://github.com/oksk1111/vit_test.git
cd vit_test

# 의존성 설치
pip install -r requirements.txt

# 또는 가상환경 사용 (권장)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

> 💡 **여러 컴퓨터에서 작업하시나요?** [보안 토큰 관리 가이드](SECURITY.md)를 확인하세요!

## 🎯 주요 기능 사용법

### 1. Jupyter Notebook 실행

이 프로젝트는 주로 Jupyter Notebook을 통해 사용됩니다.

```bash
# 가상환경 활성화
source .venv/bin/activate  # 또는 ./activate_env.sh

# Jupyter Notebook 실행
jupyter notebook cifar10_bird_dog_classification.ipynb
```

### 2. 테스트 모드와 실제 모드

개발 및 디버깅을 위한 빠른 테스트 모드와 정확한 평가를 위한 실제 모드를 제공합니다.

```python
# 테스트 모드 설정 (빠른 실험용)
TEST_MODE = True

# 실제 모드 설정 (전체 데이터셋 학습 및 평가)
TEST_MODE = False
```

### 3. 체크포인트 관리

체크포인트 파일의 자동 관리 및 복구 기능을 제공합니다.

```python
# 모든 체크포인트 검증
verify_all_checkpoints(checkpoint_dir="model_checkpoints", fix_corrupted=True)

# 특정 체크포인트에서 학습 재개
checkpoint_data = load_checkpoint_for_training(model, checkpoint_path, optimizer, scheduler)
```

## 📊 모델 평가 및 성능

이 프로젝트는 다양한 모델 아키텍처의 성능을 비교합니다:

| 모델 | 정확도 | 특징 |
|------|--------|------|
| Text Only | ~65-70% | 텍스트 특성만 사용 |
| CNN Image | ~85-90% | 표준 CNN 아키텍처 |
| ViT Only | ~90-95% | Vision Transformer 기반 |
| Early Fusion | ~92-96% | 이미지+텍스트 초기 융합 |
| Late Fusion | ~91-95% | 이미지+텍스트 후기 융합 |
| Attention Fusion | ~93-97% | 어텐션 기반 동적 융합 |

## 🛠️ 프로젝트 구조

```
vit_test/
├── cifar10_bird_dog_classification.ipynb  # 메인 노트북
├── model_checkpoints/                     # 체크포인트 저장 디렉토리
│   ├── vit_only/
│   ├── text_only/
│   └── ...
├── models/                                # 최종 모델 저장 디렉토리
├── data/                                  # CIFAR-10 데이터셋
└── requirements.txt                       # 필요 패키지

## 🔧 주요 구현 상세

### 모델 아키텍처

1. **텍스트 전용 모델 (TextOnlyModel)**
   - 텍스트 특성을 이용한 간단한 MLP 구조
   - 다양한 텍스트 특성을 융합하여 분류에 활용

2. **CNN 이미지 모델 (CNNImageModel)**
   - ResNet18 또는 VGG16 기반 구조
   - 마지막 레이어를 새로운 분류 헤드로 대체

3. **ViT 전용 모델 (ViTOnlyModel)**
   - Hugging Face의 `google/vit-base-patch16-224` 기반
   - 분류 헤드 교체로 CIFAR-10 이미지에 최적화

4. **멀티모달 융합 모델들**
   - **조기 융합 (EarlyFusionModel)**: 특성 추출 전 이미지와 텍스트 결합
   - **후기 융합 (LateFusionModel)**: 각각 특성 추출 후 결합
   - **어텐션 융합 (AttentionFusionModel)**: 어텐션 메커니즘을 통한 동적 결합

### 체크포인트 자동 관리

- **체크포인트 검증**: 체크포인트 파일의 무결성 확인
- **손상 복구**: 손상된 파일 감지 및 백업
- **자동 재개**: 중단된 학습을 마지막 체크포인트에서 재개

## ⚙️ 고급 기능

### 테스트 모드 사용

```python
# 테스트 모드 설정
TEST_MODE = True
SAMPLES_PER_CLASS = 100  # 클래스당 샘플 수 제한
```

### 하이퍼파라미터 조정

```python
# 학습률 조정
learning_rate = 5e-5  # 기본 학습률
fusion_lr = 1e-4      # 융합 모델용 학습률

# 배치 크기 조정
std_batch_size = 64   # 일반 모델용
vit_batch_size = 32   # ViT 기반 모델용
```

## 🚨 주의사항

1. **메모리 사용량**: ViT 기반 모델은 메모리 사용량이 높을 수 있음
2. **학습 시간**: 전체 데이터셋으로 학습 시 시간이 오래 걸릴 수 있음
3. **체크포인트 구조**: 이중 디렉토리 구조(`model_checkpoints/모델명/모델명/`)에 체크포인트 저장

## 🤝 기여

이 프로젝트에 기여하려면:

1. 이슈를 생성하여 변경사항 논의
2. Fork 및 feature branch 생성
3. 변경사항 구현 및 테스트
4. Pull Request 생성

## 📝 라이선스

MIT License

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.
