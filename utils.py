import os
import random
import numpy as np
import torch
import shutil

def set_seed(seed=42):
    """실험 재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_files(directory):
    """디렉터리 내 파일 개수 반환"""
    return len(os.listdir(directory))

def prepare_experiment_dir(experiment_name):
    """실험 디렉터리 초기화"""
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
