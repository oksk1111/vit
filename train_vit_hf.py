"""
Hugging Face ViT를 활용한 이미지 분류 학습 코드
CIFAR-10 데이터셋을 사용하여 ViT 모델을 fine-tuning 합니다.
의료 데이터로 교체하여 재학습할 수 있도록 설계되었습니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
from typing import Dict, List, Tuple
import json

class ViTImageClassifier:
    """
    Hugging Face ViT를 활용한 이미지 분류기
    """
    
    def __init__(
        self, 
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 10,
        output_dir: str = "./results"
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        
        # 프로세서와 모델 초기화
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"모델이 {self.device}에 로드되었습니다.")
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_dataset(self, dataset_name: str = "cifar10") -> Tuple[Dataset, Dataset]:
        """
        데이터셋을 로드하고 전처리합니다.
        
        Args:
            dataset_name: 사용할 데이터셋 이름 (기본값: cifar10)
            
        Returns:
            train_dataset, test_dataset
        """
        print(f"{dataset_name} 데이터셋을 로드 중...")
        
        # CIFAR-10 데이터셋 로드
        dataset = load_dataset(dataset_name)
        
        # 클래스 이름 저장
        self.class_names = dataset["train"].features["label"].names
        print(f"클래스: {self.class_names}")
        
        def preprocess_images(examples):
            """이미지 전처리 함수"""
            images = [img.convert("RGB") for img in examples["img"]]
            inputs = self.processor(images, return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs
        
        # 데이터셋 전처리
        train_dataset = dataset["train"].with_transform(preprocess_images)
        test_dataset = dataset["test"].with_transform(preprocess_images)
        
        print(f"훈련 데이터: {len(train_dataset)} 샘플")
        print(f"테스트 데이터: {len(test_dataset)} 샘플")
        
        return train_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        save_steps: int = 500
    ):
        """
        모델을 훈련합니다.
        
        Args:
            train_dataset: 훈련 데이터셋
            eval_dataset: 평가 데이터셋
            num_epochs: 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            save_steps: 저장 간격
        """
        print("훈련을 시작합니다...")
        
        # 훈련 인자 설정
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            eval_steps=save_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # wandb 등 비활성화
            remove_unused_columns=False,
        )
        
        # Trainer 초기화
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # 훈련 실행
        trainer.train()
        
        # 최종 평가
        eval_results = trainer.evaluate()
        print(f"최종 평가 결과: {eval_results}")
        
        # 모델 저장
        trainer.save_model()
        self.processor.save_pretrained(self.output_dir)
        
        print(f"모델이 {self.output_dir}에 저장되었습니다.")
        
        return trainer
    
    def predict(self, image_path: str) -> Dict:
        """
        단일 이미지에 대한 예측을 수행합니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            예측 결과 딕셔너리
        """
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 전처리
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 결과 처리
        predicted_class_idx = predictions.argmax().item()
        confidence = predictions.max().item()
        
        result = {
            "predicted_class": self.class_names[predicted_class_idx],
            "predicted_class_idx": predicted_class_idx,
            "confidence": confidence,
            "all_probabilities": predictions.cpu().numpy().tolist()[0]
        }
        
        return result
    
    def evaluate_test_set(self, test_dataset: Dataset) -> Dict:
        """
        테스트 셋에 대한 전체 평가를 수행합니다.
        """
        print("테스트 셋 평가 중...")
        
        # Trainer로 평가
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics,
        )
        
        eval_results = trainer.evaluate(test_dataset)
        
        # 상세 예측 결과
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # 분류 보고서
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("=== 분류 보고서 ===")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return {
            "eval_results": eval_results,
            "classification_report": report,
            "predictions": y_pred.tolist(),
            "true_labels": y_true.tolist()
        }
    
    def visualize_predictions(self, test_dataset: Dataset, num_samples: int = 8):
        """
        예측 결과를 시각화합니다.
        """
        print("예측 결과 시각화 중...")
        
        # 샘플 선택
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # 데이터 가져오기
            sample = test_dataset[idx]
            image = sample["pixel_values"]
            true_label = sample["labels"]
            
            # 이미지를 PIL로 변환 (시각화용)
            # ViTImageProcessor가 정규화를 적용하므로 역변환
            mean = np.array(self.processor.image_mean)
            std = np.array(self.processor.image_std)
            
            # 텐서를 numpy로 변환하고 정규화 해제
            img_array = image.squeeze().permute(1, 2, 0).numpy()
            img_array = img_array * std + mean
            img_array = np.clip(img_array, 0, 1)
            
            # 예측 수행
            with torch.no_grad():
                inputs = {"pixel_values": image.unsqueeze(0).to(self.device)}
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = predictions.argmax().item()
                confidence = predictions.max().item()
            
            # 시각화
            axes[i].imshow(img_array)
            axes[i].set_title(
                f"True: {self.class_names[true_label]}\n"
                f"Pred: {self.class_names[predicted_class]}\n"
                f"Conf: {confidence:.3f}",
                fontsize=10
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/predictions_visualization.png", dpi=150)
        plt.show()
        
        print(f"시각화 결과가 {self.output_dir}/predictions_visualization.png에 저장되었습니다.")

def main():
    """
    메인 실행 함수
    """
    print("=== Hugging Face ViT 이미지 분류 시작 ===")
    
    # 분류기 초기화
    classifier = ViTImageClassifier(
        model_name="google/vit-base-patch16-224",
        num_labels=10,  # CIFAR-10
        output_dir="./vit_classifier_results"
    )
    
    # 데이터셋 준비
    train_dataset, test_dataset = classifier.prepare_dataset("cifar10")
    
    # 소규모 서브셋으로 빠른 테스트 (전체 데이터셋 사용시 주석 해제)
    print("빠른 테스트를 위해 서브셋 사용 중...")
    train_subset = train_dataset.select(range(1000))  # 1000개 샘플만 사용
    test_subset = test_dataset.select(range(200))    # 200개 샘플만 사용
    
    # 훈련 실행
    trainer = classifier.train(
        train_dataset=train_subset,
        eval_dataset=test_subset,
        num_epochs=2,
        batch_size=8,
        learning_rate=2e-5
    )
    
    # 테스트 셋 평가
    eval_results = classifier.evaluate_test_set(test_subset)
    
    # 예측 시각화
    classifier.visualize_predictions(test_subset)
    
    # 결과 저장
    results_summary = {
        "model_name": classifier.model_name,
        "num_labels": classifier.num_labels,
        "class_names": classifier.class_names,
        "eval_results": eval_results["eval_results"],
        "accuracy": eval_results["classification_report"]["accuracy"]
    }
    
    with open(f"{classifier.output_dir}/training_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("=== 훈련 완료 ===")
    print(f"최종 정확도: {eval_results['classification_report']['accuracy']:.4f}")
    print(f"결과는 {classifier.output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
