import argparse
from ultralytics import YOLO
from config import BEST_MODEL_PATH_TEMPLATE

def test_model(plane, dataset, split="val", rect=False):
    """학습된 모델의 성능 테스트
    
    Args:
        plane (str): 평가할 평면 (예: 'axial', 'coronal', 'sagittal').
        dataset (str): 사용할 데이터셋 이름 (예: 'base', 'preproc', 'vit_sorted').
        split (str, optional): 평가할 데이터셋 분할 (기본값: 'val').
        rect (bool, optional): 직사각형 평가 모드를 사용할지 여부 (기본값: False).
    """
    best_model_path = BEST_MODEL_PATH_TEMPLATE.format(plane=plane, dataset=dataset)
    experiment_name = f"test/{plane}_{dataset}"

    # 모델 로드 및 평가
    model = YOLO(best_model_path)
    model.val(
        name=experiment_name,
        split=split,
        save_txt=True,
        save_conf=True,
        conf=0.25 if split == "val" else None,
        rect=rect  # 사용 여부 지정
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained YOLO model on specific dataset and plane.")
    parser.add_argument('--plane', type=str, required=True, help="The plane to use (e.g., 'axial', 'coronal', 'sagittal').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to use (e.g., 'base', 'preproc', 'vit_sorted').")
    parser.add_argument('--split', type=str, default="val", help="The dataset split to evaluate (default: 'val').")
    parser.add_argument('--rect', action='store_true', help="Use rectangular evaluation mode if set (default: False).")

    args = parser.parse_args()
    
    test_model(args.plane, args.dataset, args.split, args.rect)
