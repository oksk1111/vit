#!/usr/bin/env python3
"""
ViT 이미지 분류기 실행 스크립트
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="ViT 이미지 분류기")
    parser.add_argument(
        "--mode", 
        choices=["train", "predict", "validate"],
        default="train",
        help="실행 모드"
    )
    parser.add_argument(
        "--data", 
        type=str,
        help="데이터셋 경로 (디렉토리 또는 CSV 파일)"
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="google/vit-base-patch16-224",
        help="사용할 ViT 모델명"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="./results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        default=3,
        help="훈련 에포크 수"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=16,
        help="배치 크기"
    )
    parser.add_argument(
        "--image", 
        type=str,
        help="예측할 이미지 경로 (predict 모드용)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 훈련 모드
        if args.data:
            print("커스텀 데이터셋으로 훈련을 시작합니다...")
            # TODO: 커스텀 데이터셋 훈련 로직 구현
            print("커스텀 데이터셋 훈련은 medical_data_utils.py를 참조하세요.")
        else:
            print("CIFAR-10 데이터셋으로 훈련을 시작합니다...")
            from train_vit_hf import main as train_main
            train_main()
    
    elif args.mode == "predict":
        if not args.image:
            print("예측 모드에서는 --image 인자가 필요합니다.")
            sys.exit(1)
        
        print(f"이미지 예측: {args.image}")
        # TODO: 예측 로직 구현
        
    elif args.mode == "validate":
        if not args.data:
            print("검증 모드에서는 --data 인자가 필요합니다.")
            sys.exit(1)
        
        print(f"데이터셋 검증: {args.data}")
        from medical_data_utils import DatasetConverter
        results = DatasetConverter.validate_dataset(data_dir=args.data)
        
        print("=== 검증 결과 ===")
        print(f"유효함: {results['valid']}")
        if results['errors']:
            print("오류:")
            for error in results['errors']:
                print(f"  - {error}")
        if results['warnings']:
            print("경고:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        if results['stats']:
            print("통계:")
            for key, value in results['stats'].items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
