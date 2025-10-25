import json
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.vector_store import RecipeVectorStore
from src.models.recipe import Recipe

def load_recipes(file_path: str) -> list[Recipe]:
    """Load recipes from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes_data = json.load(f)
    
    recipes = []
    for recipe_data in recipes_data:
        try:
            recipe = Recipe(**recipe_data)
            recipes.append(recipe)
        except Exception as e:
            print(f"Error parsing recipe {recipe_data.get('title', 'unknown')}: {e}")
    
    return recipes

def format_recipe_text(recipe: Recipe) -> str:
    """Format recipe into searchable text"""
    text = f"{recipe.title}\n\n"
    
    if recipe.description:
        text += f"{recipe.description}\n\n"
    
    if recipe.cuisine_type:
        text += f"Cuisine: {recipe.cuisine_type}\n"
    if recipe.diet_type:
        text += f"Diet: {recipe.diet_type}\n"
    
    text += "\nIngredients:\n"
    for ing in recipe.ingredients:
        # Build quantity string
        if ing.quantity_g is not None:
            amount = f"{ing.quantity_g}g"
        elif ing.quantity_ml is not None:
            amount = f"{ing.quantity_ml}ml"
        elif ing.quantity_pieces is not None:
            amount = f"{ing.quantity_pieces} pieces"
        elif ing.quantity and ing.quantity_unit:
            amount = f"{ing.quantity} {ing.quantity_unit}"
        elif ing.quantity:
            amount = ing.quantity
        else:
            amount = ""
            
        text += f"- {amount} {ing.ingredient_name}"
        if ing.notes:
            text += f" ({ing.notes})"
        text += "\n"
    
    text += "\nInstructions:\n"
    for idx, step in enumerate(recipe.cooking_steps, 1):
        text += f"{idx}. {step}\n"
    
    if recipe.notes:
        text += "\nNotes:\n" + "\n".join(f"- {note}" for note in recipe.notes)
    
    return text

def main():
    print("Initializing recipe database...")
    
    # Initialize vector store
    db = RecipeVectorStore()
    
    # Load recipes
    recipes_path = project_root / "real_recipes_10.json"
    recipes = load_recipes(str(recipes_path))
    print(f"Loaded {len(recipes)} recipes")
    
    # Add recipes to vector store
    for recipe in recipes:
        recipe_text = format_recipe_text(recipe)
        metadata = {
            "title": recipe.title,
            "cuisine_type": recipe.cuisine_type or "",
            "diet_type": recipe.diet_type or "",
            "calories": str(recipe.calories_per_serving) if recipe.calories_per_serving else "",
            "difficulty": recipe.difficulty_level or "",
            "prep_time": str(recipe.prep_time_minutes) if recipe.prep_time_minutes else "",
            "cook_time": str(recipe.cooking_time_minutes) if recipe.cooking_time_minutes else "",
            "servings": str(recipe.servings) if recipe.servings else "",
            "protein": str(recipe.protein_g) if recipe.protein_g else "",
            "carbs": str(recipe.carbohydrates_g) if recipe.carbohydrates_g else "",
            "fat": str(recipe.fat_g) if recipe.fat_g else ""
        }
        
        try:
            db.add_recipe(recipe.id, recipe_text, metadata)
            print(f"Added recipe: {recipe.title}")
        except Exception as e:
            print(f"Error adding recipe {recipe.title}: {e}")

if __name__ == "__main__":
    main()