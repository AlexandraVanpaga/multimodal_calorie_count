import os

# BASE_DIR = r'C:\Users\Alexandra\Desktop\multimodal_calorie_count'
BASE_DIR = '/home/ubuntu/multimodal_calorie_count'

DATA_DIR = os.path.join(BASE_DIR, 'data', 'nutrition', 'data')

PATHS = {
    # Сырые данные
    'raw_data': os.path.join(BASE_DIR, 'data', 'nutrition.zip'),
    
    # Разархивированные данные (корневая папка)
    'extracted_data': os.path.join(BASE_DIR, 'data', 'nutrition'),
    
    # Папка data внутри nutrition
    'data_dir': DATA_DIR,
    
    # CSV файлы
    'ingredients_csv': os.path.join(DATA_DIR, 'ingredients.csv'),
    'dish_csv': os.path.join(DATA_DIR, 'dish.csv'),
    
    # Папка с изображениями
    'images_dir': os.path.join(DATA_DIR, 'images'),

    # Обработанные данные
    'processed_dir': os.path.join(BASE_DIR, 'data', 'processed'),
    'processed_dish': os.path.join(BASE_DIR, 'data', 'processed', 'dish_dataset.csv'),

    # Модели
    'models_dir': os.path.join(BASE_DIR, 'models'),
    'best_model': os.path.join(BASE_DIR, 'models', 'best_calorie_model.pth'),
    'checkpoint': os.path.join(BASE_DIR, 'models', 'checkpoint.pth'),
    
    # Логи
    'logs_dir': os.path.join(BASE_DIR, 'logs'),
    'training_log': os.path.join(BASE_DIR, 'logs', 'training_history.csv'),
    'training_viz': os.path.join(BASE_DIR, 'logs', 'training_visualization.png'),

}