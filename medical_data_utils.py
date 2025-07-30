"""
의료 데이터셋 교체를 위한 유틸리티 코드
커스텀 데이터셋을 쉽게 통합할 수 있도록 설계되었습니다.
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import pandas as pd

class CustomMedicalDataset(Dataset):
    """
    의료 이미지 데이터셋을 위한 커스텀 데이터셋 클래스
    
    디렉토리 구조:
    data/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── class2/
    │       ├── image3.jpg
    │       └── image4.jpg
    └── test/
        ├── class1/
        └── class2/
    
    또는 CSV 파일 형태:
    data.csv:
    image_path,label,split
    /path/to/image1.jpg,0,train
    /path/to/image2.jpg,1,test
    """
    
    def __init__(
        self, 
        data_dir: str = None,
        csv_file: str = None,
        split: str = "train",
        processor=None,
        class_names: List[str] = None
    ):
        """
        Args:
            data_dir: 이미지 데이터가 있는 디렉토리 경로
            csv_file: 이미지 경로와 라벨이 있는 CSV 파일
            split: 'train' 또는 'test'
            processor: Hugging Face 이미지 프로세서
            class_names: 클래스 이름 리스트
        """
        self.processor = processor
        self.split = split
        
        if csv_file:
            self.data = self._load_from_csv(csv_file, split)
        elif data_dir:
            self.data = self._load_from_directory(data_dir, split)
        else:
            raise ValueError("data_dir 또는 csv_file 중 하나는 제공되어야 합니다.")
        
        # 클래스 이름 설정
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = list(set([item['label_name'] for item in self.data]))
            self.class_names.sort()
        
        # 라벨을 숫자로 매핑
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"{split} 데이터셋 로드 완료: {len(self.data)} 샘플")
        print(f"클래스: {self.class_names}")
    
    def _load_from_csv(self, csv_file: str, split: str) -> List[Dict]:
        """CSV 파일에서 데이터 로드"""
        df = pd.read_csv(csv_file)
        df_split = df[df['split'] == split]
        
        data = []
        for _, row in df_split.iterrows():
            data.append({
                'image_path': row['image_path'],
                'label_name': str(row['label']),
                'label_idx': None  # 나중에 설정
            })
        
        return data
    
    def _load_from_directory(self, data_dir: str, split: str) -> List[Dict]:
        """디렉토리 구조에서 데이터 로드"""
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists():
            raise ValueError(f"디렉토리가 존재하지 않습니다: {split_dir}")
        
        data = []
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        data.append({
                            'image_path': str(img_file),
                            'label_name': class_name,
                            'label_idx': None  # 나중에 설정
                        })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 로드
        image = Image.open(item['image_path']).convert('RGB')
        
        # 라벨 인덱스 설정
        if item['label_idx'] is None:
            item['label_idx'] = self.label_to_idx[item['label_name']]
        
        # 프로세서가 있으면 전처리 적용
        if self.processor:
            processed = self.processor(image, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(),
                'labels': item['label_idx']
            }
        else:
            return {
                'image': image,
                'labels': item['label_idx']
            }

class DatasetConverter:
    """
    다양한 의료 데이터셋 형태를 표준 형태로 변환하는 유틸리티
    """
    
    @staticmethod
    def create_csv_from_directory(data_dir: str, output_csv: str, train_ratio: float = 0.8):
        """
        디렉토리 구조에서 CSV 파일 생성
        
        Args:
            data_dir: 데이터 디렉토리 경로
            output_csv: 출력 CSV 파일 경로
            train_ratio: 훈련 데이터 비율
        """
        data_rows = []
        
        data_path = Path(data_dir)
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                images = []
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        images.append(str(img_file))
                
                # 훈련/테스트 분할
                import random
                random.shuffle(images)
                train_size = int(len(images) * train_ratio)
                
                for i, img_path in enumerate(images):
                    split = "train" if i < train_size else "test"
                    data_rows.append({
                        'image_path': img_path,
                        'label': class_name,
                        'split': split
                    })
        
        # CSV 저장
        df = pd.DataFrame(data_rows)
        df.to_csv(output_csv, index=False)
        
        print(f"CSV 파일이 생성되었습니다: {output_csv}")
        print(f"총 {len(data_rows)} 개의 이미지")
        print(f"훈련: {len(df[df['split'] == 'train'])} 개")
        print(f"테스트: {len(df[df['split'] == 'test'])} 개")
    
    @staticmethod
    def validate_dataset(data_dir: str = None, csv_file: str = None) -> Dict:
        """
        데이터셋 유효성 검사
        
        Returns:
            검사 결과 딕셔너리
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if csv_file:
            # CSV 파일 검사
            try:
                df = pd.read_csv(csv_file)
                required_columns = ['image_path', 'label', 'split']
                
                for col in required_columns:
                    if col not in df.columns:
                        results['valid'] = False
                        results['errors'].append(f"필수 컬럼 누락: {col}")
                
                # 이미지 파일 존재 여부 확인
                missing_files = []
                for _, row in df.iterrows():
                    if not os.path.exists(row['image_path']):
                        missing_files.append(row['image_path'])
                
                if missing_files:
                    results['warnings'].append(f"{len(missing_files)}개의 이미지 파일이 존재하지 않습니다.")
                
                # 통계
                results['stats'] = {
                    'total_images': len(df),
                    'train_images': len(df[df['split'] == 'train']),
                    'test_images': len(df[df['split'] == 'test']),
                    'classes': df['label'].unique().tolist(),
                    'class_distribution': df['label'].value_counts().to_dict()
                }
                
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"CSV 파일 읽기 오류: {str(e)}")
        
        elif data_dir:
            # 디렉토리 구조 검사
            data_path = Path(data_dir)
            
            if not data_path.exists():
                results['valid'] = False
                results['errors'].append(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
                return results
            
            train_dir = data_path / "train"
            test_dir = data_path / "test"
            
            if not train_dir.exists():
                results['valid'] = False
                results['errors'].append("train 디렉토리가 존재하지 않습니다.")
            
            if not test_dir.exists():
                results['warnings'].append("test 디렉토리가 존재하지 않습니다.")
            
            # 클래스별 이미지 수 확인
            class_stats = {}
            for split_dir in [train_dir, test_dir]:
                if split_dir.exists():
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            img_count = len(list(class_dir.glob("*.jpg")) + 
                                          list(class_dir.glob("*.jpeg")) + 
                                          list(class_dir.glob("*.png")))
                            
                            key = f"{split_dir.name}_{class_dir.name}"
                            class_stats[key] = img_count
            
            results['stats']['class_distribution'] = class_stats
        
        return results

def create_medical_training_config(
    dataset_path: str,
    model_name: str = "google/vit-base-patch16-224",
    num_classes: int = None,
    class_names: List[str] = None,
    output_dir: str = "./medical_vit_results"
) -> Dict:
    """
    의료 데이터 훈련을 위한 설정 생성
    
    Args:
        dataset_path: 데이터셋 경로 (디렉토리 또는 CSV 파일)
        model_name: 사용할 ViT 모델명
        num_classes: 클래스 수
        class_names: 클래스 이름 리스트
        output_dir: 결과 저장 디렉토리
        
    Returns:
        훈련 설정 딕셔너리
    """
    config = {
        "model_config": {
            "model_name": model_name,
            "num_labels": num_classes,
            "output_dir": output_dir
        },
        "dataset_config": {
            "dataset_path": dataset_path,
            "class_names": class_names
        },
        "training_config": {
            "num_epochs": 10,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "save_steps": 500,
            "eval_steps": 500
        },
        "data_augmentation": {
            "horizontal_flip": True,
            "rotation": 15,
            "brightness": 0.2,
            "contrast": 0.2
        }
    }
    
    return config

if __name__ == "__main__":
    # 예제 사용법
    print("=== 의료 데이터셋 유틸리티 예제 ===")
    
    # 설정 파일 생성 예제
    config = create_medical_training_config(
        dataset_path="./medical_data",
        num_classes=3,
        class_names=["normal", "pneumonia", "covid19"]
    )
    
    with open("medical_training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("의료 데이터 훈련 설정이 생성되었습니다: medical_training_config.json")
