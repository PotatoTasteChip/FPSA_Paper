import torch
import os
import multiprocessing
from ultralytics import YOLO
import shutil
import os
if __name__ == '__main__':
    # GPU 설정 (여러 GPU 사용 가능)
    epochs = 1500
    batch = 16
    imgsz = 640
    device = '0'  # 0번과 1번 GPU를 사용하도록 명시적으로 지정
    data_name = "4875"
    experiment_name = "experiment_test"
    config_name = "test"
    model_name = "yolo11m"
    model = YOLO(f'./configs/{config_name}.yaml').load(f'../pretrained_model/{model_name}.pt')

    try:
        # Linux 
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # Windows
        n_cpus = os.cpu_count() or multiprocessing.cpu_count()

    worker = max(1, n_cpus - 1)          # 최소 1개는 남겨두기
    print(f"Using {worker} dataloader workers")

    # manual setting for dataloader workers
    worker = worker

    # 데이터 증강 설정
    results = model.train(
        data=f'./Dataset/{data_name}/data.yaml',
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        resume= True,
        workers=worker,
        device=device,           # 여러 GPU 지정
        amp=True,                # Mixed Precision (AMP) 활성화
        save=True,
        save_period=5,
        project=f'./experiments/runs/{experiment_name}',    # TensorBoard 로그가 저장될 디렉터리 경로 지정
        name=f'train',          # 실험 이름 지정 (해당 디렉터리에 저장됨)
        flipud=0.0,              # 수직 반전 확률
        fliplr=0.5,              # 수평 반전 확률
        degrees=10.0,            # 회전 각도 범위
        scale=0.5,               # 확대/축소 비율
        translate=0.1,           # 이동 비율
        shear=2.0,               # 왜곡 비율
        hsv_h=0.015,             # 색상 변화 범위
        hsv_s=0.7,               # 채도 변화 범위
        hsv_v=0.4,               # 밝기 변화 범위
        patience=100,  # Early Stopping을 위한 patience 설정 (예: 50 에포크)
    )