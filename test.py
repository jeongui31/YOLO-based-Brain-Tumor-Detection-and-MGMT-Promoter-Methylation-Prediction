import argparse
from ultralytics import YOLO
from config import BEST_MODEL_PATH_TEMPLATE

def test_model(plane, dataset, split="val", rect=False):
    """학습된 모델의 성능 테스트"""
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
        rect=rect  # 사용자 지정 rect 파라미터 사용
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained YOLO model on specific dataset and plane.")
    parser.add_argument('--plane', type=str, required=True, help="The plane to use (e.g., 'axial', 'coronal', 'sagittal').")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to use (e.g., 'base', 'preproc', 'vit_sorted').")
    parser.add_argument('--split', type=str, default="val", help="The dataset split to evaluate (default: 'val').")
    parser.add_argument('--rect', action='store_true', help="Use rectangular evaluation mode if set (default: False).")

    args = parser.parse_args()
    
    test_model(args.plane, args.dataset, args.split, args.rect)
