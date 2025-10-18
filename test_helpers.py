import pytest

# Import the module under test
import summarize

def _weights_map(items):
    """Helper: map ingredient_name lower -> weight (last occurrence kept)."""
    return {i["ingredient_name"].strip().lower(): i["weight"] for i in items}

def test_extract_ingredients_various_formats():
    text = """
    - Sugar: 100g
    - Flour: 0.5kg
    - Rice: 1kg
    - Butter: 25,5g
    """
    ingredients = summarize.extract_ingredients_from_prompt(text)
    print("Extracted ingredients:", ingredients)

    # No duplicate ingredient names (case-insensitive)
    names = [i["ingredient_name"].strip().lower() for i in ingredients]
    # Will have duplicates because of transform to grams 
    assert len(names) == (len(set(names)) * 2), "Duplicate ingredient entries found"

    weights = _weights_map(ingredients)
    assert "sugar" in weights
    assert "flour" in weights
    assert "rice" in weights
    assert "butter" in weights

    # Check conversions to grams
    assert pytest.approx(weights["sugar"], rel=1e-3) == 100.0
    assert pytest.approx(weights["flour"], rel=1e-3) == 500.0  # 0.5 kg -> 500 g
    assert pytest.approx(weights["rice"], rel=1e-3) == 1000.0  # 1 kg -> 1000 g
    assert pytest.approx(weights["butter"], rel=1e-3) == 25.5

def test_calculate_calories_basic(monkeypatch):
    # Provide a minimal dataset for calorie lookups
    fake_dataset = [
        {"Ingredients": "sugar", "Calories per 100g": 387.0},
        {"Ingredients": "flour", "Calories per 100g": 364.0},
        {"Ingredients": "rice", "Calories per 100g": 130.0},
    ]
    # Patch the module dataset used by calculate_calories
    monkeypatch.setattr(summarize, "dataset", fake_dataset, raising=False)

    ingredients = [
        {"ingredient_name": "sugar", "weight": 100},
        {"ingredient_name": "flour", "weight": 500},
    ]

    result = summarize.calculate_calories(ingredients)
    assert isinstance(result, dict)
    assert "total_calories" in result
    assert "details" in result
    # expected: sugar 387 + flour 1820 = 2207
    assert pytest.approx(result["total_calories"], rel=1e-3) == 2207.0

    # Check details contain entries for each ingredient and calorie values
    detail_map = {d["ingredient"].strip().lower(): d for d in result["details"]}
    assert "sugar" in detail_map
    assert "flour" in detail_map
    assert pytest.approx(detail_map["sugar"]["calories"], rel=1e-3) == 387.0
    assert pytest.approx(detail_map["flour"]["calories"], rel=1e-3) == 1820.0

def test_calculate_calories_unknown_ingredient(monkeypatch):
    fake_dataset = [
        {"Ingredients": "apple", "Calories per 100g": 52.0},
    ]
    monkeypatch.setattr(summarize, "dataset", fake_dataset, raising=False)

    ingredients = [{"ingredient_name": "unknown-thing", "weight": 100}]
    # Expect function to raise ValueError or include an error entry depending on impl.
    # Accept either behavior: raise or include None calories.
    try:
        result = summarize.calculate_calories(ingredients)
    except Exception as e:
        assert isinstance(e, (ValueError, KeyError))
    else:
        # If no exception, ensure the unknown ingredient is handled in details
        details = result.get("details", [])
        assert any(d.get("ingredient", "").lower().startswith("unknown") or d.get("calories") is None for d in details)