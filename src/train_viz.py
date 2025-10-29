# visualization.py
"""
Визуализация результатов обучения модели оценки калорийности
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import PATHS


def plot_training_history(history_csv_path, save_path=None):
    """
    Создание визуализации истории обучения
    
    Args:
        history_csv_path: путь к CSV файлу с историей обучения
        save_path: путь для сохранения графика (по умолчанию рядом с CSV)
    
    Returns:
        fig: matplotlib figure объект
    """
    # Загрузка данных
    history_df = pd.read_csv(history_csv_path)
    
    # Настройка стиля
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Loss (MSE)
    axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], 
                    label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history_df['epoch'], history_df['val_loss'], 
                    label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Loss (Mean Squared Error)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE (главная метрика)
    axes[0, 1].plot(history_df['epoch'], history_df['train_mae'], 
                    label='Train MAE', marker='o', linewidth=2, color='#2ecc71')
    axes[0, 1].plot(history_df['epoch'], history_df['val_mae'], 
                    label='Val MAE', marker='s', linewidth=2, color='#e74c3c')
    axes[0, 1].axhline(y=50, color='red', linestyle='--', linewidth=2, 
                       label='Целевой MAE (50 ккал)')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE (ккал)', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. R² Score
    axes[1, 0].plot(history_df['epoch'], history_df['train_r2'], 
                    label='Train R²', marker='o', linewidth=2, color='#3498db')
    axes[1, 0].plot(history_df['epoch'], history_df['val_r2'], 
                    label='Val R²', marker='s', linewidth=2, color='#9b59b6')
    axes[1, 0].axhline(y=0.7, color='green', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label='Хороший порог (0.7)')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² (Coefficient of Determination)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Percentile Metrics
    axes[1, 1].plot(history_df['epoch'], history_df['val_mae_under_30'], 
                    label='<30 ккал', marker='o', linewidth=2)
    axes[1, 1].plot(history_df['epoch'], history_df['val_mae_under_50'], 
                    label='<50 ккал', marker='s', linewidth=2)
    axes[1, 1].plot(history_df['epoch'], history_df['val_mae_under_100'], 
                    label='<100 ккал', marker='^', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Процент предсказаний (%)', fontsize=12)
    axes[1, 1].set_title('Точность предсказаний (% с ошибкой меньше порога)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Train vs Val MAE (сравнение)
    epochs = history_df['epoch']
    x = range(len(epochs))
    width = 0.35
    axes[2, 0].bar([i - width/2 for i in x], history_df['train_mae'], 
                   width, label='Train MAE', alpha=0.8)
    axes[2, 0].bar([i + width/2 for i in x], history_df['val_mae'], 
                   width, label='Val MAE', alpha=0.8)
    axes[2, 0].set_xlabel('Epoch', fontsize=12)
    axes[2, 0].set_ylabel('MAE (ккал)', fontsize=12)
    axes[2, 0].set_title('Train vs Val MAE', fontsize=14, fontweight='bold')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(epochs)
    axes[2, 0].legend(fontsize=11)
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # 6. Итоговая таблица с метриками
    best_epoch = history_df.loc[history_df['val_mae'].idxmin()]
    axes[2, 1].axis('off')
    table_data = [
        ['Метрика', 'Значение'],
        ['─' * 30, '─' * 20],
        ['Лучший Epoch', f"{int(best_epoch['epoch'])}"],
        ['Val MAE', f"{best_epoch['val_mae']:.2f} ккал"],
        ['Val R²', f"{best_epoch['val_r2']:.4f}"],
        ['Val MSE', f"{best_epoch['val_mse']:.2f}"],
        ['─' * 30, '─' * 20],
        ['<30 ккал', f"{best_epoch['val_mae_under_30']:.1f}%"],
        ['<50 ккал', f"{best_epoch['val_mae_under_50']:.1f}%"],
        ['<100 ккал', f"{best_epoch['val_mae_under_100']:.1f}%"],
        ['─' * 30, '─' * 20]
    ]
    
    table = axes[2, 1].table(cellText=table_data, cellLoc='left', loc='center',
                             colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Стилизация заголовка таблицы
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Стилизация разделителей
    for row in [1, 6, 10]:
        for col in range(2):
            table[(row, col)].set_facecolor('#ecf0f1')
    
    axes[2, 1].set_title('Лучшие метрики', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Сохранение
    if save_path is None:
        save_dir = os.path.dirname(history_csv_path)
        save_path = os.path.join(save_dir, 'training_visualization.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Визуализация сохранена: {save_path}")
    
    return fig


def visualize_training(log_dir=None):
    """
    Создание визуализации из директории с логами
    
    Args:
        log_dir: директория с логами (по умолчанию PATHS['logs_dir'])
    """
    if log_dir is None:
        log_dir = PATHS['logs_dir']
    
    history_path = os.path.join(log_dir, 'training_history.csv')
    
    if not os.path.exists(history_path):
        print(f"❌ Файл не найден: {history_path}")
        return None
    
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ")
    print("=" * 80 + "\n")
    
    fig = plot_training_history(history_path)
    
    # Вывод лучших метрик
    history_df = pd.read_csv(history_path)
    best_epoch = history_df.loc[history_df['val_mae'].idxmin()]
    
    print("\n" + "=" * 80)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"Epoch: {int(best_epoch['epoch'])}")
    print(f"Val MAE: {best_epoch['val_mae']:.2f} ккал")
    print(f"Val R²: {best_epoch['val_r2']:.4f}")
    print(f"<50 ккал: {best_epoch['val_mae_under_50']:.1f}%")
    print("=" * 80 + "\n")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Визуализация результатов обучения
    visualize_training()