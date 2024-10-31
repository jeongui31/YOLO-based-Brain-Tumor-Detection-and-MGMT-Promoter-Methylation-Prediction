import os

# 각 데이터셋의 이미지 및 라벨 경로
DATA_PATHS = {
    "axial":{
        "base":"data/axial/base",
        "preproc":"data/axial/contrast",
        "bg":"data/axial/contrast_bg",
        "vit_sorted":"sorted_data/axial"
    },
    "coronal":{
        "base":"data/coronal/base",
        "preproc":"data/coronal/cropHE",
        "bg":"data/coronal/cropHE_bg",
        "vit_sorted":"sorted_data/coronal"
    },
    "sagittal":{
        "base":"data/sagittal/base",
        "preproc":"data/sagittal/sharpening",
        "bg":"data/sagittal/sharpening_bg",
        "vit_sorted":"sorted_data/sagittal"
    }
}

# 데이터셋별 YAML 파일 경로
YAML_FILES = {
    "axial":{
        "base":"config/axial/axial_base.yaml",
        "preproc":"config/axial/axial_preproc.yaml",
        "bg":"config/axial/axial_preproc_bg.yaml",
        "vit_sorted":"config/axial/axial_vit_sorted.yaml"
    },
    "coronal":{
        "base":"config/coronal/coronal_base.yaml",
        "preproc":"config/coronal/coronal_preproc.yaml",
        "bg":"config/coronal/coronal_preproc_bg.yaml",
        "vit_sorted":"config/coronal/coronal_vit_sorted.yaml"
    },
    "sagittal":{
        "base":"config/sagittal/sagittal_base.yaml",
        "preproc":"config/sagittal/sagittal_preproc.yaml",
        "bg":"config/sagittal/sagittal_preproc_bg.yaml",
        "vit_sorted":"config/sagittal/sagittal_vit_sorted.yaml"
    }
}

# YOLO 모델 설정
MODEL_PATH = './yolov8n.pt'
BEST_MODEL_PATH_TEMPLATE = "runs/detect/train/{plane}_{dataset}/weights/best.pt"
