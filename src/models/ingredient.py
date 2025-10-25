from pydantic import BaseModel, Field
from typing import Optional

class IngredientData(BaseModel):
    name: str
    calories_per_100g: float
    category: str
    note: Optional[str] = None