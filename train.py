import os
import cv2
import copy
import torch
import shutil
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
from utils import set_seed, prepare_experiment_dir
from config import YAML_FILES, MODEL_PATH, BEST_MODEL_PATH_TEMPLATE
from sklearn.metrics import accuracy_score

# Seed 설정
set_seed(42)

# argparse로 인수 받아오기
def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT and YOLO models with custom epochs")
    parser.add_argument("--vit_epochs", type=int, default=25, help="Number of epochs for ViT training")
    parser.add_argument("--yolo_epochs", type=int, default=200, help="Number of epochs for YOLO training")
    return parser.parse_args()

# 데이터셋 클래스 정의
class BrainDataset(Dataset):
    def __init__(self, image_dir, shape_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.shape_dir = shape_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        shape_path = os.path.join(self.shape_dir, img_name.replace('.jpg', '.txt'))
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert('RGB')
        with open(shape_path, 'r') as file:
            shape = file.readline().strip()
        
        with open(label_path, 'r') as file:
            label = file.readline().strip()
        
        if self.transform:
            image = self.transform(image)
        
        shape_map = {'axial': 0, 'coronal': 1, 'sagittal': 2}
        shape = shape_map[shape]  
        
        return image, shape, label, img_name

# Transformer 모델 학습 함수 정의
def train_vit_model(model, dataloader, dataset_size, device, num_epochs=25):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        
        model.train()  # 학습 모드 설정
        
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # 매개변수 경사도 초기화
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()  # 역전파
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', end="\n\n")

        # 정확도 향상 시 모델 가중치 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best Train Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)  # 최적의 모델 로드
    return model

# 평가 함수 정의
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels, _, _ in dataloader: 
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')

# YOLO 모델 학습 함수 정의
def train_yolo_model(plane, dataset="vit_sorted", device="0", yolo_epochs=200):
    yaml_file = YAML_FILES[plane][dataset]
    experiment_name = f"train/{plane}_{dataset}"

    # 모델 학습
    model = YOLO(MODEL_PATH)
    model.train(
        data=yaml_file,
        epochs=yolo_epochs,
        device=device,
        name=experiment_name,
        exist_ok=True,
        save_txt=True,
        save_conf=True
    )

# 바운딩 박스 로드 및 저장 함수
def load_bboxes(label_file):
    bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height, int(class_id)])
    return bboxes

def save_yolo_bboxes(bboxes, file_path):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            class_id = bbox[4]
            x_center, y_center, width, height = bbox[:4]
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# 전처리 함수 정의
def process_axial(image_path, label_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)
    bboxes = load_bboxes(label_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return False

    image = cv2.convertScaleAbs(image, alpha=1.5, beta=-50)    
    cv2.imwrite(output_image_path, image)
    save_yolo_bboxes(bboxes, output_label_path)
    return True

has_copied_coronal_images = False
def process_coronal(image_path, label_path, output_image_path, output_label_path, ratio=0.05, seed=42):
    global has_copied_coronal_images  # 복사 작업 확인을 위한 전역 변수

    # 이미지 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(cropped_gray, 60, 255, cv2.THRESH_BINARY)
        masked_img = cv2.bitwise_and(cropped_gray, cropped_gray, mask=mask)
        img_hist_eq = cv2.equalizeHist(masked_img)
        img_hist_eq = img_hist_eq * (mask > 0) + cropped_gray * (mask == 0)
        
        # 전처리된 이미지 저장
        cv2.imwrite(output_image_path, img_hist_eq)

        # 바운딩 박스 변환 및 저장
        bboxes = load_bboxes(label_path)
        new_bboxes = []

        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            x_center = x_center * image.shape[1] - x
            y_center = y_center * image.shape[0] - y
            x_center /= w
            y_center /= h
            width *= image.shape[1] / w
            height *= image.shape[0] / h
            new_bboxes.append([x_center, y_center, width, height, class_id])

        save_yolo_bboxes(new_bboxes, output_label_path)
        
        # 첫 번째 호출 시에만 5% 이미지 복사
        if not has_copied_coronal_images:
            processed_folder = os.path.dirname(output_image_path)  # 이미지가 저장된 폴더
            random.seed(seed)
            all_processed_images = [f for f in os.listdir(processed_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_images_to_copy = int(len(all_processed_images) * ratio)
            selected_images = random.sample(all_processed_images, num_images_to_copy)

            for img in selected_images:
                src_path = os.path.join(processed_folder, img)
                dst_path = os.path.join(processed_folder, f"copy_{img}")
                if not os.path.exists(dst_path):  # 중복 방지
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")
            
            has_copied_coronal_images = True  # 복사 작업이 완료되었음을 표시

        return True
    else:
        print(f"No contours found in image: {image_path}")
        return False


def process_sagittal(image_path, label_path, output_image_path, output_label_path):
    image = cv2.imread(image_path)
    bboxes = load_bboxes(label_path)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    sharpened_image_pil = image_pil.filter(ImageFilter.SHARPEN)
    sharpened_img = cv2.cvtColor(np.array(sharpened_image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, sharpened_img)
    save_yolo_bboxes(bboxes, output_label_path)
    return True

# 데이터 분류 함수 정의
def classify(vit_model, train_loader, test_loader, device, base_path):
    vit_model.eval()
    class_map = {0: 'axial', 1: 'coronal', 2: 'sagittal'}

    # train 데이터 분류
    with torch.no_grad():
        for inputs, _, _, img_names in train_loader:
            inputs = inputs.to(device)
            outputs = vit_model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for img_name, pred in zip(img_names, preds):
                class_name = class_map[pred.item()]
                
                # 이미지 및 라벨 경로 정의
                img_src = os.path.join(train_loader.dataset.image_dir, img_name)
                label_src = os.path.join(train_loader.dataset.label_dir, img_name.replace('.jpg', '.txt'))

                # 저장할 이미지 및 라벨 경로
                img_dst = os.path.join(base_path, class_name, 'vit_sorted', 'images', 'train', img_name)
                label_dst = os.path.join(base_path, class_name, 'vit_sorted', 'labels', 'train', img_name.replace('.jpg', '.txt'))

                # 디렉토리 생성
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                os.makedirs(os.path.dirname(label_dst), exist_ok=True)

                # 전처리 함수 선택 및 적용
                if class_name == 'axial':
                    process_axial(img_src, label_src, img_dst, label_dst)
                elif class_name == 'coronal':
                    process_coronal(img_src, label_src, img_dst, label_dst)
                elif class_name == 'sagittal':
                    process_sagittal(img_src, label_src, img_dst, label_dst)

    # test 데이터 분류
    with torch.no_grad():
        for inputs, _, _, img_names in test_loader:
            inputs = inputs.to(device)
            outputs = vit_model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for img_name, pred in zip(img_names, preds):
                class_name = class_map[pred.item()]
                
                # 이미지 및 라벨 경로 정의
                img_src = os.path.join(test_loader.dataset.image_dir, img_name)
                label_src = os.path.join(test_loader.dataset.label_dir, img_name.replace('.jpg', '.txt'))

                # 저장할 이미지 및 라벨 경로
                img_dst = os.path.join(base_path, class_name, 'vit_sorted', 'images', 'test', img_name)
                label_dst = os.path.join(base_path, class_name, 'vit_sorted', 'labels', 'test', img_name.replace('.jpg', '.txt'))

                # 디렉토리 생성
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                os.makedirs(os.path.dirname(label_dst), exist_ok=True)

                # 전처리 함수 선택 및 적용
                if class_name == 'axial':
                    process_axial(img_src, label_src, img_dst, label_dst)
                elif class_name == 'coronal':
                    process_coronal(img_src, label_src, img_dst, label_dst)
                elif class_name == 'sagittal':
                    process_sagittal(img_src, label_src, img_dst, label_dst)

# YOLO 모델 학습 함수 정의
def train_yolo(base_path, yolo_epochs):
    shape_names = ['axial', 'coronal', 'sagittal']

    # 정렬된 데이터로 YOLO 모델 학습
    for shape_name in shape_names:
        print(f"Training YOLO for class: {shape_name}")
        train_yolo_model(plane=shape_name, dataset="vit_sorted", yolo_epochs=yolo_epochs)

# 메인 함수
def main():
    # 인수 파싱
    args = parse_args()
    
    prepare_experiment_dir("experiments/vit")
    prepare_experiment_dir("data/axial/vit_sorted")
    prepare_experiment_dir("data/coronal/vit_sorted")
    prepare_experiment_dir("data/sagittal/vit_sorted")

    base_path = './data'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 데이터셋과 데이터 로더 준비
    train_dataset = BrainDataset(
        image_dir=os.path.join(base_path, 'total', 'images/train'),
        shape_dir=os.path.join(base_path, 'total', 'shapes/train'),
        label_dir=os.path.join(base_path, 'total', 'labels/train'),
        transform=data_transforms['train']
    )
    test_dataset = BrainDataset(
        image_dir=os.path.join(base_path, 'total', 'images/test'),
        shape_dir=os.path.join(base_path, 'total', 'shapes/test'),
        label_dir=os.path.join(base_path, 'total', 'labels/test'),
        transform=data_transforms['test']
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ViT 모델 정의 및 학습
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=3).to(device)
    vit_model = train_vit_model(vit_model, train_loader, len(train_dataset), device, num_epochs=args.vit_epochs)
    torch.save(vit_model.state_dict(), 'experiments/vit/best_model.pth')

    # ViT 모델 결과
    evaluate_model(vit_model, test_loader, device)
    
    # ViT 모델로 train과 test 분류
    classify(vit_model, train_loader, test_loader, device, base_path)

    # YOLO 모델 학습
    train_yolo(base_path, yolo_epochs=args.yolo_epochs)

if __name__ == "__main__":
    main()




# import os
# import torch
# import shutil
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from timm import create_model
# from ultralytics import YOLO
# from PIL import Image, ImageFilter
# import cv2
# import numpy as np
# from utils import set_seed, prepare_experiment_dir
# from config import YAML_FILES, MODEL_PATH, BEST_MODEL_PATH_TEMPLATE
# import copy
# from sklearn.metrics import accuracy_score

# # Seed 설정
# set_seed(42)

# # 데이터셋 클래스 정의
# class BrainDataset(Dataset):
#     def __init__(self, image_dir, shape_dir, label_dir, transform=None):
#         self.image_dir = image_dir
#         self.shape_dir = shape_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = os.listdir(image_dir)
        
#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         shape_path = os.path.join(self.shape_dir, img_name.replace('.jpg', '.txt'))
#         label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
#         image = Image.open(img_path).convert('RGB')
#         with open(shape_path, 'r') as file:
#             shape = file.readline().strip()
        
#         with open(label_path, 'r') as file:
#             label = file.readline().strip()
        
#         if self.transform:
#             image = self.transform(image)
        
#         shape_map = {'axial': 0, 'coronal': 1, 'sagittal': 2}
#         shape = shape_map[shape]  
        
#         return image, shape, label, img_name

# # Transformer 모델 학습 함수 정의
# def train_vit_model(model, dataloader, dataset_size, device, num_epochs=25):
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print('-' * 10)

#         running_loss = 0.0
#         running_corrects = 0
        
#         model.train()  # 학습 모드 설정
        
#         for inputs, labels, _, _ in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()  # 매개변수 경사도 초기화
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)

#             loss.backward()  # 역전파
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / dataset_size
#         epoch_acc = running_corrects.double() / dataset_size

#         print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#         print("\n")

#         # 정확도 향상 시 모델 가중치 저장
#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = copy.deepcopy(model.state_dict())

#     print(f'Best Train Acc: {best_acc:.4f}')

#     model.load_state_dict(best_model_wts)  # 최적의 모델 로드
#     return model

# # 평가 함수 정의
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for inputs, labels, _, _ in dataloader: 
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     print(f'Test Accuracy: {accuracy:.4f}')

# # YOLO 모델 학습 함수 정의
# def train_yolo_model(plane, dataset="vit_sorted", device="0"):
#     yaml_file = YAML_FILES[plane][dataset]
#     experiment_name = f"train/{plane}_{dataset}"

#     # 모델 학습
#     model = YOLO(MODEL_PATH)
#     model.train(
#         data=yaml_file,
#         epochs=200,
#         device=device,
#         name=experiment_name,
#         exist_ok=True,
#         save_txt=True,
#         save_conf=True
#     )

# # YOLO 모델용 전처리 함수 정의
# def normalize_bbox(xyxy, img_width, img_height):
#     x_center = (xyxy[0] + xyxy[2]) / 2.0 / img_width
#     y_center = (xyxy[1] + xyxy[3]) / 2.0 / img_height
#     width = (xyxy[2] - xyxy[0]) / img_width
#     height = (xyxy[3] - xyxy[1]) / img_height
#     return [x_center, y_center, width, height]

# # 바운딩 박스 로드 및 저장 함수
# def load_bboxes(label_file):
#     bboxes = []
#     with open(label_file, 'r') as f:
#         for line in f:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             bboxes.append([x_center, y_center, width, height, int(class_id)])
#     return bboxes

# def save_yolo_bboxes(bboxes, file_path):
#     with open(file_path, 'w') as f:
#         for bbox in bboxes:
#             class_id = bbox[4]
#             x_center, y_center, width, height = bbox[:4]
#             f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# # 전처리 함수 정의
# def process_axial(image_path, label_path, output_image_path, output_label_path):
#     image = cv2.imread(image_path)
#     bboxes = load_bboxes(label_path)
#     if image is None:
#         print(f"Could not read image: {image_path}")
#         return False

#     image = cv2.convertScaleAbs(image, alpha=1.5, beta=-50)    
#     cv2.imwrite(output_image_path, image)
#     save_yolo_bboxes(bboxes, output_label_path)
#     return True

# def process_coronal(image_path, label_path, output_image_path, output_label_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Could not read image: {image_path}")
#         return False

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cropped_image = image[y:y+h, x:x+w]
#         cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#         _, mask = cv2.threshold(cropped_gray, 60, 255, cv2.THRESH_BINARY)
#         masked_img = cv2.bitwise_and(cropped_gray, cropped_gray, mask=mask)
#         img_hist_eq = cv2.equalizeHist(masked_img)
#         img_hist_eq = img_hist_eq * (mask > 0) + cropped_gray * (mask == 0)
#         cv2.imwrite(output_image_path, img_hist_eq)

#         bboxes = load_bboxes(label_path)
#         new_bboxes = []

#         for bbox in bboxes:
#             x_center, y_center, width, height, class_id = bbox
#             x_center = x_center * image.shape[1] - x
#             y_center = y_center * image.shape[0] - y
#             x_center /= w
#             y_center /= h
#             width *= image.shape[1] / w
#             height *= image.shape[0] / h
#             new_bboxes.append([x_center, y_center, width, height, class_id])

#         save_yolo_bboxes(new_bboxes, output_label_path)
#         return True
#     else:
#         print(f"No contours found in image: {image_path}")
#         return False

# def process_sagittal(image_path, label_path, output_image_path, output_label_path):
#     image = cv2.imread(image_path)
#     bboxes = load_bboxes(label_path)
#     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     sharpened_image_pil = image_pil.filter(ImageFilter.SHARPEN)
#     sharpened_img = cv2.cvtColor(np.array(sharpened_image_pil), cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_image_path, sharpened_img)
#     save_yolo_bboxes(bboxes, output_label_path)
#     return True

# # 데이터 분류 함수 정의
# def classify(vit_model, train_loader, test_loader, device, base_path):
#     vit_model.eval()
#     class_map = {0: 'axial', 1: 'coronal', 2: 'sagittal'}

#     # train 데이터 분류
#     with torch.no_grad():
#         for inputs, _, _, img_names in train_loader:
#             inputs = inputs.to(device)
#             outputs = vit_model(inputs)
#             _, preds = torch.max(outputs, 1)
            
#             for img_name, pred in zip(img_names, preds):
#                 class_name = class_map[pred.item()]
                
#                 # 이미지 및 라벨 경로 정의
#                 img_src = os.path.join(train_loader.dataset.image_dir, img_name)
#                 label_src = os.path.join(train_loader.dataset.label_dir, img_name.replace('.jpg', '.txt'))

#                 # 저장할 이미지 및 라벨 경로
#                 img_dst = os.path.join(base_path, class_name, 'vit_sorted', 'images', 'train', img_name)
#                 label_dst = os.path.join(base_path, class_name, 'vit_sorted', 'labels', 'train', img_name.replace('.jpg', '.txt'))

#                 # 디렉토리 생성
#                 os.makedirs(os.path.dirname(img_dst), exist_ok=True)
#                 os.makedirs(os.path.dirname(label_dst), exist_ok=True)

#                 # 전처리 함수 선택 및 적용
#                 if class_name == 'axial':
#                     process_axial(img_src, label_src, img_dst, label_dst)
#                 elif class_name == 'coronal':
#                     process_coronal(img_src, label_src, img_dst, label_dst)
#                 elif class_name == 'sagittal':
#                     process_sagittal(img_src, label_src, img_dst, label_dst)

#     # test 데이터 분류
#     with torch.no_grad():
#         for inputs, _, _, img_names in test_loader:
#             inputs = inputs.to(device)
#             outputs = vit_model(inputs)
#             _, preds = torch.max(outputs, 1)
            
#             for img_name, pred in zip(img_names, preds):
#                 class_name = class_map[pred.item()]
                
#                 # 이미지 및 라벨 경로 정의
#                 img_src = os.path.join(test_loader.dataset.image_dir, img_name)
#                 label_src = os.path.join(test_loader.dataset.label_dir, img_name.replace('.jpg', '.txt'))

#                 # 저장할 이미지 및 라벨 경로
#                 img_dst = os.path.join(base_path, class_name, 'vit_sorted', 'images', 'test', img_name)
#                 label_dst = os.path.join(base_path, class_name, 'vit_sorted', 'labels', 'test', img_name.replace('.jpg', '.txt'))

#                 # 디렉토리 생성
#                 os.makedirs(os.path.dirname(img_dst), exist_ok=True)
#                 os.makedirs(os.path.dirname(label_dst), exist_ok=True)

#                 # 전처리 함수 선택 및 적용
#                 if class_name == 'axial':
#                     process_axial(img_src, label_src, img_dst, label_dst)
#                 elif class_name == 'coronal':
#                     process_coronal(img_src, label_src, img_dst, label_dst)
#                 elif class_name == 'sagittal':
#                     process_sagittal(img_src, label_src, img_dst, label_dst)

# # YOLO 모델 학습 함수 정의
# def train_yolo(base_path):
#     shape_names = ['axial', 'coronal', 'sagittal']

#     # 정렬된 데이터로 YOLO 모델 학습
#     for shape_name in shape_names:
#         print(f"Training YOLO for class: {shape_name}")
#         train_yolo_model(plane=shape_name, dataset="vit_sorted")

# # 메인 함수
# def main():
#     prepare_experiment_dir("experiments/vit")
#     prepare_experiment_dir("data/axial/vit_sorted")
#     prepare_experiment_dir("data/coronal/vit_sorted")
#     prepare_experiment_dir("data/sagittal/vit_sorted")

#     base_path = './data'
#     transformer_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # 데이터셋과 데이터 로더 준비
#     train_dataset = BrainDataset(
#         image_dir=os.path.join(base_path, 'total', 'images/train'),
#         shape_dir=os.path.join(base_path, 'total', 'shapes/train'),
#         label_dir=os.path.join(base_path, 'total', 'labels/train'),
#         transform=transformer_transforms
#     )
#     test_dataset = BrainDataset(
#         image_dir=os.path.join(base_path, 'total', 'images/test'),
#         shape_dir=os.path.join(base_path, 'total', 'shapes/test'),
#         label_dir=os.path.join(base_path, 'total', 'labels/test'),
#         transform=transformer_transforms
#     )
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#     # ViT 모델 정의 및 학습
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     vit_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=3).to(device)
#     vit_model = train_vit_model(vit_model, train_loader, len(train_dataset), device, num_epochs=25)
#     torch.save(vit_model.state_dict(), 'experiments/vit/best_model.pth')

#     # ViT 모델 결과
#     evaluate_model(vit_model, test_loader, device)
    
#     # ViT 모델로 train과 test 분류
#     classify(vit_model, train_loader, test_loader, device, base_path)

#     # YOLO 모델 학습
#     train_yolo(base_path)

# if __name__ == "__main__":
#     main()

# from ultralytics import YOLO
# from config import DATA_PATHS, YAML_FILES, MODEL_PATH
# from utils import set_seed, prepare_experiment_dir

# def train_model(plane, dataset):
#     """지정한 단면과 데이터셋에 대해 YOLO 모델을 학습"""
#     yaml_file = YAML_FILES[plane][dataset]
#     experiment_name = f"train/{plane}_{dataset}"

#     # 모델 학습
#     model = YOLO(MODEL_PATH)
#     model.train(
#         data=yaml_file,
#         epochs=3,
#         device=0,
#         name=experiment_name,
#         exist_ok=True,
#         save_txt=True,
#         save_conf=True
#     )

# if __name__ == "__main__":
#     set_seed(42)
#     train_model("axial", "base")
    
