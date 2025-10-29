# utils.py
"""
Утилиты для обучения модели оценки калорийности:
- Dataset и DataLoader
- Архитектура модели
- Функция обучения
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def set_requires_grad(module: nn.Module, unfreeze_pattern: str = "", verbose: bool = False):
    """Заморозка/разморозка слоёв"""
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


# ============================================================================
# DATASET
# ============================================================================

class CalorieDataset(Dataset):
    """Dataset для загрузки изображений блюд с описанием ингредиентов"""
    
    def __init__(self, dataframe, images_dir, config, tokenizer, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Проверка наличия изображений
        print("Проверка наличия изображений...")
        missing = []
        for dish_id in self.data['dish_id']:
            img_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
            if not os.path.exists(img_path):
                missing.append(dish_id)
        
        if missing:
            print(f"Внимание: {len(missing)} изображений не найдено")
            print(f"Примеры отсутствующих: {missing[:5]}")
            self.data = self.data[~self.data['dish_id'].isin(missing)].reset_index(drop=True)
            print(f"Датасет очищен: осталось {len(self.data)} примеров")
        else:
            print(f"Все {len(self.data)} изображений найдены")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Загрузка изображения
        img_name = row['dish_id']
        img_path = os.path.join(self.images_dir, img_name, 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Токенизация текста
        ingredients = row['dish_description']
        tokens = self.tokenizer(
            ingredients,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Масса и калории
        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'mass': mass,
            'calories': calories,
            'dish_id': img_name
        }


def get_transforms(image_size=224):
    """Получить трансформации для train и test"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def prepare_dataloaders(dish_dataset, config):
    """
    Подготовка DataLoader'ов для обучения и валидации
    
    Args:
        dish_dataset: датафрейм с данными о блюдах
        config: конфигурация обучения
    
    Returns:
        train_loader, test_loader: загрузчики данных
    """
    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Разделение данных
    train_df = dish_dataset[dish_dataset['split'] == 'train'].copy()
    test_df = dish_dataset[dish_dataset['split'] == 'test'].copy()
    
    # Трансформации
    train_transform, test_transform = get_transforms()
    
    # Создание датасетов
    train_dataset = CalorieDataset(train_df, config.IMAGES_DIR, config, tokenizer, transform=train_transform)
    test_dataset = CalorieDataset(test_df, config.IMAGES_DIR, config, tokenizer, transform=test_transform)
    
    # Создание загрузчиков
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nTrain: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"Test: {len(test_loader)} batches ({len(test_dataset)} samples)\n")
    
    return train_loader, test_loader


# ============================================================================
# MODEL
# ============================================================================

class CalorieEstimationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Text encoder (BERT)
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        # Image encoder
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        
        # Projection layers
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)
        
        # Regression head 
        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT / 2),
            nn.Linear(config.HIDDEN_DIM // 4, 1),
            nn.ReLU()
        )
        
    def forward(self, image, ingredients, mass):
        # Эмбеддинги
        text_features = self.text_model(**ingredients).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)
        
        # Проекция
        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj(mass)
        
        # Early fusion - простое перемножение (РАБОТАЕТ ЛУЧШЕ!)
        fused_emb = text_emb * image_emb * mass_emb
        
        # Предсказание
        calories = self.regressor(fused_emb)
        return calories


# ============================================================================
# TRAINING
# ============================================================================

def train(config, device, train_loader, test_loader):
    """
    Функция обучения модели
    
    Args:
        config: конфигурация обучения
        device: устройство (CPU/CUDA)
        train_loader: загрузчик тренировочных данных
        test_loader: загрузчик тестовых данных
    
    Returns:
        history: история обучения
    """
    print(f"\n{'='*80}")
    print(f"ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
    print(f"{'='*80}\n")
    
    # Создание папок
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
    os.makedirs(config.LOG_PATH, exist_ok=True)
    
    # Модель
    model = CalorieEstimationModel(config).to(device)
    
    # Заморозка слоёв
    print("Настройка заморозки слоёв...")
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)
    
    # Оптимизатор
    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.text_proj.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.image_proj.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.mass_proj.parameters(), 'lr': config.REGRESSOR_LR}
    ])
    
    criterion = nn.MSELoss()
    
    # История
    history = {
        'epoch': [], 'train_loss': [], 'train_mae': [], 'train_mse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_mse': [], 'val_r2': [],
        'val_mae_under_30': [], 'val_mae_under_50': [], 'val_mae_under_100': []
    }
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*80}\n")
    
    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]"):
            images = batch['image'].to(device)
            ingredients = {'input_ids': batch['input_ids'].to(device), 
                          'attention_mask': batch['attention_mask'].to(device)}
            mass = batch['mass'].to(device).unsqueeze(1)
            calories = batch['calories'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(images, ingredients, mass)
            loss = criterion(predictions, calories)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(predictions.detach().cpu().numpy().flatten())
            train_targets.extend(calories.detach().cpu().numpy().flatten())
        
        train_loss = np.mean(train_losses)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_mse = mean_squared_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
                images = batch['image'].to(device)
                ingredients = {'input_ids': batch['input_ids'].to(device),
                              'attention_mask': batch['attention_mask'].to(device)}
                mass = batch['mass'].to(device).unsqueeze(1)
                calories = batch['calories'].to(device).unsqueeze(1)
                
                predictions = model(images, ingredients, mass)
                loss = criterion(predictions, calories)
                
                val_losses.append(loss.item())
                val_preds.extend(predictions.cpu().numpy().flatten())
                val_targets.extend(calories.cpu().numpy().flatten())
        
        val_loss = np.mean(val_losses)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_mse = mean_squared_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)
        
        errors = np.abs(np.array(val_targets) - np.array(val_preds))
        mae_under_30 = (errors < 30).mean() * 100
        mae_under_50 = (errors < 50).mean() * 100
        mae_under_100 = (errors < 100).mean() * 100
        
        # Сохранение истории
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_mse'].append(train_mse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)
        history['val_mae_under_30'].append(mae_under_30)
        history['val_mae_under_50'].append(mae_under_50)
        history['val_mae_under_100'].append(mae_under_100)
        
        # Вывод
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train | Loss: {train_loss:.4f} | MAE: {train_mae:.2f} ккал | R²: {train_r2:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | MAE: {val_mae:.2f} ккал | R²: {val_r2:.4f}")
        print(f"Val   | <30: {mae_under_30:.1f}% | <50: {mae_under_50:.1f}% | <100: {mae_under_100:.1f}%")
        
        # Сохранение лучшей модели
        if val_mae < best_val_mae - config.MIN_DELTA:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config,
                'history': history
            }, config.SAVE_PATH)
            print(f"Лучшая модель сохранена (MAE: {val_mae:.2f} ккал)")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.PATIENCE}")
        
        if patience_counter >= config.PATIENCE:
            print(f"\nРанняя остановка на epoch {epoch+1}")
            break
        
        print(f"{'─'*80}\n")
    
    # Сохранение истории
    history_df = pd.DataFrame(history)
    log_path = os.path.join(config.LOG_PATH, 'training_history.csv')
    history_df.to_csv(log_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*80}")
    print(f"История: {log_path}")
    print(f"Лучший Val MAE: {best_val_mae:.2f} ккал")
    print(f"{'='*80}\n")
    
    return history