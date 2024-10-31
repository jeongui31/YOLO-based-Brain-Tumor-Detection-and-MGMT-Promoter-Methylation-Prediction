import os
from PIL import Image

def load_dataset(plane, dataset_name):
    """
    Args:
        plane (str): 단면명 (e.g., 'axial', 'coronal', 'sagittal').
        dataset_name (str): 데이터셋 이름 (e.g., 'dataset1', 'dataset2').

    Returns:
        List[Image.Image]: 모든 이미지를 포함한 리스트.
    """
    data_path = f"./data/{plane}/{dataset_name}"
    if not os.path.exists(data_path):
        raise ValueError(f"경로가 존재하지 않습니다: {data_path}")

    # 폴더 내 모든 이미지 파일 불러오기
    image_files = sorted(os.listdir(data_path))
    images = [Image.open(os.path.join(data_path, img)).convert('RGB') for img in image_files]
    
    return images
