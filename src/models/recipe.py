from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid

class RecipeIngredient(BaseModel):
    ingredient_name: str
    quantity_g: Optional[float] = None
    quantity_ml: Optional[float] = None
    quantity_pieces: Optional[int] = None
    quantity_unit: Optional[str] = None
    quantity: Optional[str] = None
    notes: Optional[str] = None

class Recipe(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    ingredients: List[RecipeIngredient]
    cooking_steps: List[str]
    cuisine_type: Optional[str] = None
    diet_type: Optional[str] = None
    prep_time_minutes: Optional[int] = None
    cooking_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    difficulty_level: Optional[str] = None
    calories_per_serving: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    rating: Optional[float] = None
    tags: Optional[List[str]] = None
    notes: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        # Allow both 'title' and 'name' to be used
        field_renames = {'name': 'title'}