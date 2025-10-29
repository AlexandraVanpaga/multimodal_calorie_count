# training_config.py
"""
Конфигурация обучения мультимодальной модели для оценки калорийности
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn

from config.paths import PATHS  

def seed_everything(seed: int):
    """Установка seed для воспроизводимости результатов"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(module: nn.Module, unfreeze_pattern: str = "", verbose: bool = False):
    """
    Заморозка/разморозка слоёв нейронной сети по паттерну
    
    Args:
        module: PyTorch модуль
        unfreeze_pattern: паттерн для разморозки слоёв (через |)
        verbose: выводить названия размороженных слоёв
    """
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return
    
    pattern = unfreeze_pattern.split("|")
    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class TrainingConfig:
    """Конфигурация для обучения модели оценки калорийности"""
    
    # Воспроизводимость
    SEED = 42
    
    # Архитектура моделей
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b7"
    
    # Заморозка слоёв (пустая строка = заморозить все)
    TEXT_MODEL_UNFREEZE = "encoder.layer.10|encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры обучения
    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 1e-3
    EPOCHS = 50
    DROPOUT = 0.3
    HIDDEN_DIM = 512
    
    # Пути к данным и сохранению
    PROCESSED_DATA_PATH = PATHS['processed_dish']
    IMAGES_DIR = PATHS['images_dir']
    SAVE_PATH = PATHS['best_model']
    LOG_PATH = PATHS['logs_dir']
    
    # Ранняя остановка
    PATIENCE = 10
    MIN_DELTA = 1.0


def get_device():
    """Получить доступное устройство (CUDA/CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(dish_dataset):
    """
    Полный цикл обучения модели
    
    Args:
        dish_dataset: датафрейм с данными о блюдах
    
    Returns:
        history: история обучения
    """
    from src.utils import prepare_dataloaders, train
    
    # Инициализация
    config = TrainingConfig()
    device = get_device()
    seed_everything(config.SEED)
    
    print(f"\n{'='*80}")
    print(f"КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
    print(f"{'='*80}")
    print(f"Seed: {config.SEED}")
    print(f"Device: {device}")
    print(f"Text model: {config.TEXT_MODEL_NAME}")
    print(f"Image model: {config.IMAGE_MODEL_NAME}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"{'='*80}\n")
    
    # Подготовка данных
    train_loader, test_loader = prepare_dataloaders(dish_dataset, config)
    
    # Обучение
    history = train(config, device, train_loader, test_loader)
    
    return history


if __name__ == "__main__":
    import pandas as pd
    
    # Загрузка обработанных данных
    dish_dataset = pd.read_csv(PATHS['processed_dish'])
    
    # Запуск обучения
    history = run_training(dish_dataset)