from src.database.vector_store import RecipeVectorStore
from src.database.ingredient_db import IngredientDB
from src.models.recipe import Recipe
from typing import List, Optional, Dict, Any
import json

class RecipeService:
    def __init__(self):
        self.db = RecipeVectorStore()
        self.ingredient_db = IngredientDB()

    def add_recipe(self, recipe: Recipe):
        """Add a recipe to the vector store with calculated calories"""
        # Calculate calories per serving if not provided
        if not recipe.calories_per_serving:
            total_calories = self.calculate_calories(recipe)
            recipe.calories_per_serving = total_calories if not recipe.servings else total_calories / recipe.servings
        
        # Format recipe for vector store
        recipe_text = self._format_recipe_text(recipe)
        
        # Prepare metadata
        metadata = {
            "title": recipe.title,
            "cuisine_type": recipe.cuisine_type or "",
            "diet_type": recipe.diet_type or "",
            "calories_per_serving": str(recipe.calories_per_serving),
            "difficulty_level": recipe.difficulty_level or "",
            "prep_time_minutes": str(recipe.prep_time_minutes or ""),
            "cooking_time_minutes": str(recipe.cooking_time_minutes or ""),
            "servings": str(recipe.servings or ""),
            "protein_g": str(recipe.protein_g or ""),
            "carbohydrates_g": str(recipe.carbohydrates_g or ""),
            "fat_g": str(recipe.fat_g or "")
        }
        
        # Add to vector store
        self.db.add_recipe(
            recipe_id=recipe.id,
            recipe_text=recipe_text,
            metadata=metadata
        )

    def search_recipes(self, query: str, limit: int = 3):
        return self.db.search_recipes(query, n_results=limit)

    def calculate_calories(self, recipe: Recipe) -> float:
        """Calculate total calories for a recipe based on its ingredients"""
        if recipe.calories_per_serving and recipe.servings:
            return recipe.calories_per_serving * recipe.servings
            
        total_calories = 0
        for ingredient in recipe.ingredients:
            # Extract quantity and unit based on available fields
            quantity = None
            unit = None
            
            if ingredient.quantity_g is not None:
                quantity = str(ingredient.quantity_g)
                unit = 'g'
            elif ingredient.quantity_ml is not None:
                quantity = str(ingredient.quantity_ml)
                unit = 'ml'
            elif ingredient.quantity_pieces is not None:
                quantity = str(ingredient.quantity_pieces)
                unit = 'piece'
            elif ingredient.quantity:
                quantity = ingredient.quantity
                unit = ingredient.quantity_unit
                
            if not quantity:
                continue
                
            calories = self.ingredient_db.calculate_calories(
                ingredient.ingredient_name,
                quantity,
                unit
            )
            total_calories += calories
            
        # Calculate per serving if servings specified
        if recipe.servings and recipe.servings > 0:
            total_calories = total_calories / recipe.servings
            
        return round(total_calories, 2)

    @staticmethod
    def _format_recipe_text(recipe: Recipe) -> str:
        """Format recipe into searchable text"""
        text = [recipe.title]
        
        if recipe.description:
            text.append(recipe.description)
            
        if recipe.cuisine_type:
            text.append(f"Cuisine: {recipe.cuisine_type}")
            
        if recipe.diet_type:
            text.append(f"Diet: {recipe.diet_type}")
            
        text.append("\nIngredients:")
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
                
            text.append(f"- {amount} {ing.ingredient_name}")
            if ing.notes:
                text.append(f"  Note: {ing.notes}")
                
        text.append("\nInstructions:")
        for idx, step in enumerate(recipe.cooking_steps, 1):
            text.append(f"{idx}. {step}")
            
        if recipe.notes:
            text.append("\nNotes:")
            text.extend(f"- {note}" for note in recipe.notes)
            
        if recipe.calories_per_serving:
            text.append(f"\nCalories per serving: {recipe.calories_per_serving} kcal")
            
        return "\n".join(text)

    @staticmethod
    def _convert_to_grams(amount: str, unit: Optional[str]) -> float:
        try:
            value = float(amount)
            if not unit or unit.lower() == 'g':
                return value
            elif unit.lower() == 'kg':
                return value * 1000
            else:
                return value  # Default to original value if unit unknown
        except ValueError:
            return 0.0  # Return 0 if conversion fails