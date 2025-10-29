# clean_data.py
import pandas as pd
import os
from config import PATHS


def clean_and_process_data():
    """
    Очистка и обработка данных о блюдах и ингредиентах
    """
    print("Загрузка данных...")
    # Загрузка датасетов
    dish_dataset = pd.read_csv(PATHS['dish_csv'])
    ingr_dataset = pd.read_csv(PATHS['ingredients_csv'])
    
    print("Обработка ингредиентов...")
    # Замена deprecated ингредиентов
    ingr_dataset['ingr'] = ingr_dataset['ingr'].replace('deprecated', 'Unknown ingredient')
    
    print("Очистка колонки ingredients...")
    # Очистка колонки ingredients от лишних символов и нулей
    dish_dataset['ingredients'] = dish_dataset['ingredients'].str.replace(r'[^0-9;,]', '', regex=True)
    dish_dataset['ingredients'] = dish_dataset['ingredients'].str.replace(r'(^|;)0+(?=\d)', r'\1', regex=True)
    
    print("Создание описаний блюд...")
    # Создание словаря id -> название ингредиента
    ingr_dict = dict(zip(ingr_dataset['id'], ingr_dataset['ingr']))
    
    # Функция для замены id на названия
    def get_ingredients_names(ingredients_str):
        ids = ingredients_str.split(';')
        names = [ingr_dict.get(int(id), f'Unknown_{id}') for id in ids if id]
        return '; '.join(names)
    
    # Создание колонки с описанием
    dish_dataset['dish_description'] = dish_dataset['ingredients'].apply(get_ingredients_names)
    
    print("Замена Unknown ingredients...")
    # Замена известных Unknown ingredients
    def replace_unknown_ingredients(row):
        description = row['dish_description']
        ingredients = row['ingredients']
        
        if 'Unknown ingredient' in description:
            ids = [int(x) for x in ingredients.split(';') if x]
            replacements = []
            
            if 453 in ids:
                replacements.append('apple')
            if 458 in ids:
                replacements.append('watermelon')
            
            for replacement in replacements:
                description = description.replace('Unknown ingredient', replacement, 1)
        
        return description
    
    dish_dataset['dish_description'] = dish_dataset.apply(replace_unknown_ingredients, axis=1)
    
    print("Исправление специфических записей...")
    # Исправление конкретного блюда
    dish_dataset.loc[dish_dataset['dish_id'] == 'dish_1556575700', 'ingredients'] = '471'
    dish_dataset.loc[dish_dataset['dish_id'] == 'dish_1556575700', 'dish_description'] = 'cherry tomatoes'
    dish_dataset.loc[dish_dataset['dish_id'] == 'dish_1556575700', 'total_calories'] = 15.0
    
    print("Сохранение данных...")
    # Создание папки и сохранение
    os.makedirs(PATHS['processed_dir'], exist_ok=True)
    dish_dataset.to_csv(PATHS['processed_dish'], index=False)
    
    print(f"Датафрейм сохранён: {PATHS['processed_dish']}")
    print(f"Обработано блюд: {len(dish_dataset)}")
    print(f"Train: {len(dish_dataset[dish_dataset['split'] == 'train'])}")
    print(f"Test: {len(dish_dataset[dish_dataset['split'] == 'test'])}")
    
    return dish_dataset


if __name__ == "__main__":
    clean_and_process_data()