# ViT Test - Vision Transformer Implementation and Fine-tuning

이 프로젝트는 Hugging Face의 Vision Transformer (ViT) 모델을 활용한 이미지 분류 시스템입니다. CIFAR-10 데이터셋으로 학습하고, 추후 의료 데이터로 쉽게 교체하여 재학습할 수 있도록 설계되었습니다.

## 🚀 주요 기능

- **Hugging Face ViT 모델 활용**: `google/vit-base-patch16-224` 사전 훈련된 모델 사용
- **CIFAR-10 데이터셋 학습**: 10개 클래스 이미지 분류
- **의료 데이터 호환**: 커스텀 의료 데이터셋으로 쉽게 교체 가능
- **완전한 훈련 파이프라인**: 데이터 로딩, 훈련, 평가, 시각화
- **유연한 데이터 형태 지원**: 디렉토리 구조 또는 CSV 파일

## 📦 설치

```bash
# 저장소 클론
git clone https://github.com/oksk1111/vit_test.git
cd vit_test

# 의존성 설치
pip install -r requirements.txt
```

## 🎯 빠른 시작

### 1. CIFAR-10으로 기본 훈련

```bash
# 기본 훈련 실행
python train_vit_hf.py

# 또는 실행 스크립트 사용
python run_vit.py --mode train
```

### 2. 커스텀 실행 설정

```bash
# 더 많은 에포크로 훈련
python run_vit.py --mode train --epochs 10 --batch-size 32

# 결과 디렉토리 지정
python run_vit.py --mode train --output ./my_results
```

## 🏥 의료 데이터로 교체하기

### 1. 데이터 준비

**방법 A: 디렉토리 구조**
```
medical_data/
├── train/
│   ├── normal/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── pneumonia/
│   │   ├── image3.jpg
│   │   └── image4.jpg
│   └── covid19/
│       ├── image5.jpg
│       └── image6.jpg
└── test/
    ├── normal/
    ├── pneumonia/
    └── covid19/
```

**방법 B: CSV 파일**
```csv
image_path,label,split
/path/to/image1.jpg,normal,train
/path/to/image2.jpg,pneumonia,test
/path/to/image3.jpg,covid19,train
```

### 2. 데이터셋 검증

```bash
# 디렉토리 구조 검증
python run_vit.py --mode validate --data ./medical_data

# CSV 파일 검증
python run_vit.py --mode validate --data ./medical_data.csv
```

### 3. 의료 데이터로 훈련

```python
from medical_data_utils import CustomMedicalDataset, create_medical_training_config
from train_vit_hf import ViTImageClassifier

# 설정 생성
config = create_medical_training_config(
    dataset_path="./medical_data",
    num_classes=3,
    class_names=["normal", "pneumonia", "covid19"]
)

# 분류기 초기화
classifier = ViTImageClassifier(
    num_labels=3,
    output_dir="./medical_results"
)

# 커스텀 데이터셋 로드
train_dataset = CustomMedicalDataset(
    data_dir="./medical_data",
    split="train",
    processor=classifier.processor
)

test_dataset = CustomMedicalDataset(
    data_dir="./medical_data", 
    split="test",
    processor=classifier.processor
)

# 훈련 실행
classifier.train(train_dataset, test_dataset)
```

## 📊 결과 및 성능

훈련 완료 후 다음 파일들이 생성됩니다:

- `results/`: 훈련된 모델 파일들
- `results/training_summary.json`: 훈련 요약 정보
- `results/predictions_visualization.png`: 예측 결과 시각화
- `results/logs/`: 훈련 로그

### 예상 성능 (CIFAR-10)

- **베이스라인 정확도**: ~85-90%
- **훈련 시간**: 2-3 에포크, 약 10-15분 (GPU 사용시)
- **모델 크기**: 약 330MB

## 🔧 주요 파일 설명

- `train_vit_hf.py`: 메인 훈련 스크립트
- `medical_data_utils.py`: 의료 데이터 유틸리티
- `run_vit.py`: 통합 실행 스크립트
- `vit.py`: 원본 ViT 구현 (참고용)

## ⚙️ 고급 설정

### 하이퍼파라미터 조정

```python
# 더 큰 모델 사용
classifier = ViTImageClassifier(
    model_name="google/vit-large-patch16-224",
    num_labels=your_num_classes
)

# 훈련 설정 조정
classifier.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-5,  # 더 작은 학습률
    save_steps=200
)
```

### 데이터 증강

`medical_data_utils.py`의 `create_medical_training_config()` 함수에서 데이터 증강 설정을 조정할 수 있습니다.

## 🚨 주의사항

1. **GPU 메모리**: 배치 크기가 클 경우 GPU 메모리 부족 가능
2. **의료 데이터**: 실제 의료 데이터 사용시 개인정보보호 및 규정 준수 필요
3. **모델 검증**: 의료 분야 적용시 충분한 검증과 전문가 검토 필요

## 🤝 기여

이 프로젝트에 기여를 환영합니다! 

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 라이선스

MIT License

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.
