import chromadb
from chromadb.utils import embedding_functions
from typing import Optional
from src.config.settings import settings
from src.models.ingredient import IngredientData

class IngredientDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name="ingredients",
            embedding_function=self.embedding_fn
        )

    def add_ingredient(self, ingredient: IngredientData):
        self.collection.add(
            ids=[ingredient.name],
            documents=[ingredient.name],
            metadatas=[{
                "calories_per_100g": ingredient.calories_per_100g,
                "category": ingredient.category,
                "note": ingredient.note or ""
            }]
        )

    def find_ingredient(self, name: str, threshold: float = 0.8) -> Optional[IngredientData]:
        """Find ingredient by name using semantic search"""
        results = self.collection.query(
            query_texts=[name],
            n_results=1
        )
        
        if results and results['distances'][0][0] < threshold:
            metadata = results['metadatas'][0][0]
            return IngredientData(
                name=results['documents'][0][0],
                calories_per_100g=metadata['calories_per_100g'],
                category=metadata['category'],
                note=metadata['note']
            )
        return None

    def get_all_ingredients(self):
        return self.collection.get()

    def calculate_calories(self, ingredient_name: str, amount: str, unit: Optional[str]) -> float:
        """Calculate calories for given ingredient amount"""
        ingredient = self.find_ingredient(ingredient_name)
        if not ingredient:
            return 0.0
        
        # Convert amount to grams
        grams = self._convert_to_grams(amount, unit)
        if grams == 0:
            return 0.0
            
        return (grams / 100) * ingredient.calories_per_100g

    @staticmethod
    def _convert_to_grams(amount: str, unit: Optional[str]) -> float:
        """Convert various units to grams"""
        try:
            value = float(''.join(filter(str.isdigit, amount)))
            
            if not unit:
                return value
                
            unit = unit.lower()
            conversions = {
                'g': 1,
                'kg': 1000,
                'mg': 0.001,
                'oz': 28.35,
                'lb': 453.59,
                'cup': 236.59,  # Approximate
                'tbsp': 14.79,  # Approximate
                'tsp': 4.93,    # Approximate
            }
            
            return value * conversions.get(unit, 1)
        except (ValueError, AttributeError):
            return 0.0