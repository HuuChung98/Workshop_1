import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.ingredient_db import IngredientDB
from src.models.ingredient import IngredientData

def load_ingredients(file_path: str) -> list[IngredientData]:
    """Load ingredients from dataset.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ingredients = []
    for item in data:
        try:
            ingredient = IngredientData(
                name=item["Ingredients"],
                calories_per_100g=float(item["Calories per 100g"]),
                category=item["Category"],
                note=item.get("Note")
            )
            ingredients.append(ingredient)
        except Exception as e:
            print(f"Error parsing ingredient {item.get('Ingredients', 'unknown')}: {e}")
    
    return ingredients

def main():
    print("Initializing ingredient database...")
    
    # Initialize database
    db = IngredientDB()
    
    # Load ingredients
    ingredients_path = project_root / "dataset.json"
    ingredients = load_ingredients(str(ingredients_path))
    print(f"Loaded {len(ingredients)} ingredients")
    
    # Add ingredients to database
    for ingredient in ingredients:
        try:
            db.add_ingredient(ingredient)
            print(f"Added ingredient: {ingredient.name}")
        except Exception as e:
            print(f"Error adding ingredient {ingredient.name}: {e}")

if __name__ == "__main__":
    main()