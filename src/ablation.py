# ablation_study.py
"""
Ablation Study - анализ вклада каждой модальности в предсказание калорийности
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

from config.paths import PATHS
from src.utils import CalorieDataset, CalorieEstimationModel
from src.train import TrainingConfig  # <- ИМПОРТИРУЕМ ОРИГИНАЛЬНЫЙ CONFIG


class AblationCalorieDataset(CalorieDataset):
    """Dataset с возможностью маскирования модальностей"""
    
    def __init__(self, dataframe, images_dir, config, tokenizer, transform=None, mask_mode=None):
        super().__init__(dataframe, images_dir, config, tokenizer, transform)
        self.mask_mode = mask_mode  # None, 'image', 'text', 'mass'
        
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Маскирование модальностей
        if self.mask_mode == 'image':
            item['image'] = torch.randn_like(item['image'])
        elif self.mask_mode == 'text':
            item['input_ids'] = torch.zeros_like(item['input_ids'])
            item['attention_mask'] = torch.zeros_like(item['attention_mask'])
        elif self.mask_mode == 'mass':
            item['mass'] = torch.tensor(300.0, dtype=torch.float32)
        
        return item


def validate_ablation(model, test_loader, device, mask_mode=None):
    """
    Валидация модели с маскированием модальности
    
    Args:
        model: обученная модель
        test_loader: DataLoader для тестовых данных
        device: устройство
        mask_mode: какую модальность маскировать
    
    Returns:
        mae, r2: метрики качества
    """
    model.eval()
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Ablation: mask={mask_mode}", leave=False):
            images = batch['image'].to(device)
            ingredients = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            mass = batch['mass'].to(device).unsqueeze(1)
            calories = batch['calories'].to(device).unsqueeze(1)
            
            predictions = model(images, ingredients, mass)
            
            val_preds.extend(predictions.cpu().numpy().flatten())
            val_targets.extend(calories.cpu().numpy().flatten())
    
    mae = mean_absolute_error(val_targets, val_preds)
    r2 = r2_score(val_targets, val_preds)
    
    return mae, r2


def run_ablation_study(dish_dataset, model_path=None):
    """
    Запуск полного ablation study
    
    Args:
        dish_dataset: датафрейм с данными о блюдах
        model_path: путь к обученной модели (по умолчанию best_calorie_model.pth)
    
    Returns:
        results_df: датафрейм с результатами
    """
    print("=" * 80)
    print("ABLATION STUDY - Анализ вклада модальностей")
    print("=" * 80 + "\n")
    
    # Используем оригинальный TrainingConfig
    config = TrainingConfig()  # <- ИЗМЕНЕНО
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели
    if model_path is None:
        model_path = PATHS['best_model']
    
    model = CalorieEstimationModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # <- ДОБАВЛЕНО weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Модель загружена: {os.path.basename(model_path)}")
    print(f"Val MAE из обучения: {checkpoint['val_mae']:.2f} ккал\n")
    
    # Подготовка данных
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_df = dish_dataset[dish_dataset['split'] == 'test'].copy()
    
    # Тестирование с разными маскированиями
    results = {}
    
    print("-" * 80)
    print("ТЕСТИРОВАНИЕ")
    print("-" * 80)
    
    for mask_mode, description in [
        (None, "Все модальности"),
        ('image', "Text + Mass (без изображения)"),
        ('text', "Image + Mass (без текста)"),
        ('mass', "Image + Text (без массы)")
    ]:
        dataset = AblationCalorieDataset(
            test_df, config.IMAGES_DIR, config, tokenizer,
            transform=test_transform, mask_mode=mask_mode
        )
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        mae, r2 = validate_ablation(model, loader, device, mask_mode=mask_mode)
        
        results[mask_mode] = {'mae': mae, 'r2': r2, 'description': description}
        print(f"✓ {description:30s} | MAE: {mae:6.2f} ккал | R²: {r2:.4f}")
    
    # Вычисление вклада
    print("\n" + "=" * 80)
    print("ВКЛАД КАЖДОЙ МОДАЛЬНОСТИ (деградация при удалении)")
    print("=" * 80)
    
    mae_full = results[None]['mae']
    contributions = {
        'image': results['image']['mae'] - mae_full,
        'text': results['text']['mae'] - mae_full,
        'mass': results['mass']['mae'] - mae_full
    }
    
    print(f"Image: +{contributions['image']:6.2f} ккал деградация")
    print(f"Text:  +{contributions['text']:6.2f} ккал деградация")
    print(f"Mass:  +{contributions['mass']:6.2f} ккал деградация")
    
    # Итоговая таблица
    results_df = pd.DataFrame({
        'Модальности': [
            'Все (baseline)',
            'Text + Mass',
            'Image + Mass',
            'Image + Text'
        ],
        'Замаскировано': ['—', 'Image', 'Text', 'Mass'],
        'MAE (ккал)': [
            mae_full,
            results['image']['mae'],
            results['text']['mae'],
            results['mass']['mae']
        ],
        'R²': [
            results[None]['r2'],
            results['image']['r2'],
            results['text']['r2'],
            results['mass']['r2']
        ],
        'Деградация MAE': [
            0,
            contributions['image'],
            contributions['text'],
            contributions['mass']
        ]
    })
    
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    # Сохранение результатов
    ablation_log_path = os.path.join(PATHS['logs_dir'], 'ablation_study.csv')
    results_df.to_csv(ablation_log_path, index=False)
    print(f"\n✓ Результаты сохранены: {ablation_log_path}\n")
    
    return results_df


if __name__ == "__main__":
    # Загрузка данных
    dish_dataset = pd.read_csv(PATHS['processed_dish'])
    
    # Запуск ablation study
    results = run_ablation_study(dish_dataset)