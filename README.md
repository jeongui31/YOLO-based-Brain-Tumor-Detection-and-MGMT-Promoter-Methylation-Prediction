# YOLO-based-Brain-Tumor-Detection-and-MGMT-Promoter-Methylation-Prediction
YOLO 기반 뇌 종양 탐지와 MGMT 프로모터 메틸화 예측에 관한 연구

- local setup
  ```
  pip install -r requirements.txt
  ```
  
- train models
  ```
  python train.py --vit_epochs 25 --yolo_epochs 200
  ```
  - `--vit_epochs` : Number of epochs for training ViT (default: 25)
  - `--yolo_epochs` : Number of epochs for training YOLO (default: 200)
    
- test YOLO model
  ```
  python test.py --plane axial --dataset base --split val --rect
  ```
  - `--plane` : Specify MRI plane (axial, coronal, sagittal)
  - `--dataset` : Specify dataset type (base, preproc, bg, vit_sorted)
  - `--split` : Dataset split for evaluation (default: val)
  - `--rect` : Enable rectangular evaluation mode

resources
- [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics), the official repository for Ultralytics YOLOv8, providing models, documentation, and support for YOLO-based object detection tasks.
- [Brain Tumor Object Detection Dataset on Kaggle](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets): Dataset used for training and evaluating brain tumor detection models, with labeled images across different anatomical planes.
