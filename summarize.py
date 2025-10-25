import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_random_exponential,
)
from openai import AzureOpenAI
import tiktoken
from dotenv import load_dotenv
import re
import json

load_dotenv()

with open("dataset_with_vietnamese.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# --------------------------
# Azure OpenAI Configuration
# --------------------------

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Deployment name (not model name)

if not ENDPOINT or not API_KEY or not DEPLOYMENT:
    raise EnvironmentError(
        "Missing Azure OpenAI configuration. "
        "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT."
    )

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

# --------------------------
# Prompting
# --------------------------

SYSTEM_PROMPT = """
You are a professional cooking instructor and nutrition assistant. 
Your primary goal is to guide users through cooking processes and help them understand the nutritional value of their dishes.
When the user asks in Vietnamese, respond in Vietnamese. If the user asks in English, respond in English.

Rules and Behavior:
1. If the user requests more than 100 dishes, respond:
   “Sorry, I can only provide up to 100 dishes.”
2. If the user’s question is unrelated to cooking or nutrition, respond:
   “Sorry, I cannot support topics outside the cooking context.”

3. When providing cooking instructions, ensure to include:
    - A list of ingredients with precise measurements (normally for 2 peoples).
    - Step-by-step cooking instructions (minimum 5 steps, maximum 10 steps).
    - Allergy cautions if applicable.
    - Calculate nutritional information, especially total calories based on the ingredients provided.
    
4. Always provide well-organized, professional, and easy-to-follow explanations for both cooking and nutrition guidance.

"""


USER_PROMPT = """
        For the dish [summarize and interpret the user’s context], you may consider cooking the following 
        (suggest 3 dishes if the user has not specified any):

            Dish Name: Gà nướng muối ớt
            Flavor Profile: Mặn, cay, thơm

        When providing cooking instructions, follow this structure
            Examples: 
                Dish Name: Gà nướng muối ớt (total calories: 2200 kcal)
                Ingredients:
                    - Gà nguyên con: 1500g
                    - Muối: 8g
                    - Tiêu đen xay: 4g
                    - Ớt bột: 6g
                    - Dầu ăn: 10g

                ALWAYS list ingredients as "- Tên nguyên liệu: XXg". Convert or estimate any volumetric measure (muỗng, chén, quả, etc.) into grams so calories can be computed later.

                Instructions (at least 5 steps and no more than 10 steps):
                    Bước 1: Quay gà sạch và để ráo nước.
                    Bước 2: Trộn đều muối, tiêu, ớt bột và dầu ăn trong một bát nhỏ.
                    Bước 3: Xoa hỗn hợp gia vị lên bề mặt gà, đảm bảo thấm đều.
                    Bước 4: Để gà ướp trong 30 phút cho ngấm gia vị.
                    Bước 5: Nướng gà trong lò đã được làm nóng trước ở 200 độ C trong 1 giờ.
                (Add more steps as needed, up to Step 10)

                Caution on allergy if any
                
                Nutritional information:

        Now, please analyze the following user input or transcript carefully and generate the corresponding cooking guide:

        \"\"\"
        {chunk}
        \"\"\"
    """


STYLE_GUIDES: Dict[str, str] = {
    "executive": "Concise narrative paragraphs suitable for senior leadership.",
    "bullet": "Bulleted list with short, punchy lines and bolded section headers.",
    "action-items": "Focus on tasks: bullet list with Owner, Task, Due Date, and Status if given.",
    "detailed": "Thorough multi-section report with short paragraphs and clear headings.",
}

DEFAULT_TEMPERATURE = 0.3
MAX_CHARS_PER_CHUNK = 8_000
MAX_RETRIES = 4
MAX_OUTPUT_TOKENS = 700

LAST_GENERIC_INGREDIENTS: List[Dict[str, object]] = []
LAST_DISH_INGREDIENTS: Dict[str, List[Dict[str, object]]] = {}
LAST_DISH_NAME: Optional[str] = None

# --------------------------
# Helpers
# --------------------------


def estimate_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class SummarizeConfig:
    style: str = "executive"
    lang: str = "en"
    temperature: float = DEFAULT_TEMPERATURE


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        cut = text.rfind("\n", start, end)
        if cut == -1:
            cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + int(max_chars * 0.6):
            cut = end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut
    return chunks


class TransientOpenAIError(Exception):
    pass


function_definition = [{
    "type": "function",
    "function": {
        "name": "calculate_recipe_calories",
        "description": "Calculate total calories and nutritional information for a recipe based on its ingredients",
        "parameters": {
            "type": "object",
            "properties": {
                "recipe": {
                    "type": "object",
                    "description": "Recipe information including ingredients and quantities",
                    "properties": {
                        "title": {"type": "string"},
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ingredient_name": {"type": "string"},
                                    "quantity_g": {"type": "number", "description": "Amount in grams"},
                                    "quantity_ml": {"type": "number", "description": "Amount in milliliters"},
                                    "quantity_pieces": {"type": "integer", "description": "Number of pieces"},
                                    "quantity": {"type": "string", "description": "Generic quantity"},
                                    "quantity_unit": {"type": "string", "description": "Unit for generic quantity"},
                                    "notes": {"type": "string"}
                                },
                                "required": ["ingredient_name"]
                            }
                        },
                        "servings": {"type": "integer", "description": "Number of servings"}
                    },
                    "required": ["title", "ingredients"]
                }
            },
            "required": ["recipe"]
        }
    }
}]

# Calculate calories based on ingredient name and weight
# def calculate_calories(ingredient_name: str, weight: float) -> float:
#     """Tính tổng calories dựa trên tên nguyên liệu và trọng lượng (gram)."""
#     # Tìm nguyên liệu trong dataset
#     match = next((item for item in dataset if item["Ingredients"].lower() == ingredient_name.lower()), None)
#     if not match:
#         raise ValueError(f"Không tìm thấy nguyên liệu: {ingredient_name}")

#     cal_per_100g = match["Calories per 100g"]
#     total_cal = (weight / 100) * cal_per_100g
#     return total_cal

def calculate_recipe_calories(recipe: Dict[str, Any]) -> dict:
    """Calculate total calories and nutrition details for a recipe"""
    results: List[Dict[str, Any]] = []
    total_calories = 0.0
    
    for item in recipe["ingredients"]:
        name = str(item["ingredient_name"]).strip()
        quantity = None
        unit = None
        
        # Determine quantity and unit from available fields
        if "quantity_g" in item and item["quantity_g"]:
            quantity = float(item["quantity_g"])
            unit = "g"
        elif "quantity_ml" in item and item["quantity_ml"]:
            quantity = float(item["quantity_ml"])
            unit = "ml"
        elif "quantity_pieces" in item and item["quantity_pieces"]:
            quantity = float(item["quantity_pieces"])
            unit = "piece"
        elif "quantity" in item and item["quantity"]:
            try:
                quantity = float(''.join(filter(str.isdigit, str(item["quantity"]))))
                unit = item.get("quantity_unit", "g")
            except ValueError:
                quantity = 0.0
                
        # Find ingredient in dataset
        match = next((d for d in dataset if d["Ingredients"].lower() == name.lower()), None)
        
        if not match or not quantity:
            results.append({
                "ingredient": name,
                "quantity": quantity,
                "unit": unit,
                "calories": None,
                "note": "Ingredient not found or invalid quantity"
            })
            continue
            
        # Convert to grams if needed
        grams = quantity
        if unit == "ml":
            grams = quantity  # Approximate 1ml = 1g
        elif unit == "piece":
            grams = quantity * 100  # Rough estimate per piece
            
        # Calculate calories
        cal_per_100g = match["Calories per 100g"]
        calories = (grams / 100) * cal_per_100g
        total_calories += calories
        
        results.append({
            "ingredient": name,
            "quantity": quantity,
            "unit": unit,
            "calories": round(calories, 2),
            "calories_per_100g": cal_per_100g
        })
    
    # Calculate per serving if available
    servings = recipe.get("servings")
    if servings and servings > 0:
        calories_per_serving = total_calories / servings
    else:
        calories_per_serving = total_calories
        
    return {
        "details": results,
        "total_calories": round(total_calories, 2),
        "calories_per_serving": round(calories_per_serving, 2),
        "servings": servings
    }



def _is_transient_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        k in msg
        for k in [
            "timeout",
            "overloaded",
            "temporarily",
            "rate limit",
            "retry later",
            "service unavailable",
            "502",
            "503",
            "504",
        ]
    )

def _save_last_ingredients(ingredients: List[Dict[str, object]], dish_name: Optional[str] = None):
    global LAST_GENERIC_INGREDIENTS, LAST_DISH_INGREDIENTS, LAST_DISH_NAME
    if not ingredients:
        return

    LAST_GENERIC_INGREDIENTS = ingredients

    if dish_name:
        key = dish_name.lower().strip()
        if key:
            LAST_DISH_INGREDIENTS[key] = ingredients
            LAST_DISH_NAME = key


def _format_calorie_result(result: dict) -> str:
    """Format calorie calculation results into human-readable text"""
    lines = []
    
    # Add total calories and per serving if available
    if result.get("servings"):
        lines.append(f"Total calories: {result['total_calories']} kcal")
        lines.append(f"Calories per serving: {result['calories_per_serving']} kcal")
        lines.append(f"Servings: {result['servings']}")
    else:
        lines.append(f"Total calories: {result['total_calories']} kcal")
    
    lines.append("\nBreakdown by ingredient:")
    for item in result["details"]:
        if item.get("calories") is not None:
            lines.append(
                f"- {item['ingredient']} "
                f"({item['quantity']}{item['unit']}): "
                f"{item['calories']} kcal "
                f"({item.get('calories_per_100g', 0)} kcal/100g)"
            )
        else:
            lines.append(
                f"- {item['ingredient']} "
                f"({item['quantity']}{item['unit']}): "
                "no calorie data available"
            )
    
    return "\n".join(lines)


def _wants_calorie_total(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("calo", "calorie", "calories", "kcal"))


def _extract_dish_name_from_request(text: str) -> Optional[str]:
    global LAST_DISH_NAME

    lowered = text.lower()
    if re.search(r"\bmón\s+(này|đó)\b", lowered):
        return LAST_DISH_NAME

    match = re.search(r"\bmón\s+([a-z0-9à-ỹ\s]+)", text, flags=re.IGNORECASE)
    if match:
        dish = match.group(1)
        dish = re.split(r"[\?\.!]", dish)[0]
        dish = re.sub(r"\b(này|đó|này\s+đi|đó\s+đi)\b", "", dish, flags=re.IGNORECASE).strip()
        if dish:
            return dish

    match_en = re.search(r"calories(?:\s+for|\s+of)?\s+([a-z0-9à-ỹ\s]+)", text, flags=re.IGNORECASE)
    if match_en:
        dish = re.split(r"[\?\.!]", match_en.group(1))[0].strip()
        if dish:
            return dish

    return LAST_DISH_NAME


def _get_ingredients_for_calorie_request(text: str) -> Optional[List[Dict[str, object]]]:
    if not _wants_calorie_total(text):
        return None

    dish = _extract_dish_name_from_request(text)
    if dish:
        key = dish.lower().strip()
        if key and key in LAST_DISH_INGREDIENTS:
            return LAST_DISH_INGREDIENTS[key]
        if key:
            return None

    return LAST_GENERIC_INGREDIENTS if LAST_GENERIC_INGREDIENTS else None


def _remember_dish_ingredients(text: str, fallback: Optional[List[Dict[str, object]]] = None):
    stored = False
    sections = re.split(r"(?=Dish Name:)\s*", text)
    for section in sections:
        match = re.search(r"Dish Name:\s*(.+)", section)
        if not match:
            continue
        dish_name = match.group(1).strip()
        ingredients = extract_ingredients_from_prompt(section)
        if ingredients:
            _save_last_ingredients(ingredients, dish_name)
            stored = True

    if not stored and fallback:
        _save_last_ingredients(fallback)


def _should_skip_auto_extract(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in (
        "now, please analyze the following user input",
        "dish name: gà nướng muối ớt",
        "caution on allergy",
        "always list ingredients",
    ))


def extract_ingredients_from_prompt(prompt: str) -> List[Dict[str, object]]:
    """Trích xuất danh sách nguyên liệu và trọng lượng (gram) từ chuỗi nhập."""
    text = prompt.strip()
    colon_pattern = re.compile(r'(?:-\s*)?([\w\sÀ-ỹ]+?)[:：]\s*(\d+(?:[\.,]\d+)?)\s*(kg|g|gr|gram|grams|kilogram|kilograms)?\b', re.IGNORECASE)
    leading_pattern = re.compile(r'(?:-\s*)?(\d+(?:[\.,]\d+)?)\s*(kg|g|gr|gram|grams|kilogram|kilograms)\s+([\w\sÀ-ỹ]+)', re.IGNORECASE)

    matches: List[Tuple[str, str, str]] = []

    for match in colon_pattern.finditer(text):
        name = match.group(1)
        weight_str = match.group(2)
        unit = match.group(3) or "g"
        matches.append((name, weight_str, unit))

    for match in leading_pattern.finditer(text):
        weight_str = match.group(1)
        unit = match.group(2)
        name = match.group(3)
        matches.append((name, weight_str, unit))

    ingredients: List[Dict[str, object]] = []
    seen = set()

    for name, weight_str, unit in matches:
        cleaned_name = re.sub(r'\s*\(.*?\)\s*', '', name).strip()
        if not cleaned_name:
            continue
        key = cleaned_name.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            weight = float(weight_str.replace(",", "."))
            ingredients.append(
                {"ingredient_name": name.strip().capitalize(), "weight": weight}
            )
        except ValueError:
            continue
        unit_lower = unit.lower()
        if unit_lower.startswith("kg") or "kilogram" in unit_lower:
            weight *= 1000
        ingredients.append({
            "ingredient_name": cleaned_name,
            "weight": weight
        })

    return ingredients


# --------------------------
# Chat completion with assistant role
# --------------------------


from src.services.recipe_service import RecipeService

recipe_service = RecipeService()

@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def chat_complete(system: str, user: str, temperature: float, max_tokens: int = MAX_OUTPUT_TOKENS, raw_user_text: Optional[str] = None) -> str:
    """
    Enhanced chat completion with recipe search and calorie calculation:
    1. Search vector DB for similar recipes
    2. Let GPT choose the best match
    3. Calculate calories using ingredient DB
    """
    try:
        # Use raw user text if available, otherwise use formatted user prompt
        target_text = raw_user_text if raw_user_text is not None else user
        
        # Skip system prompts and examples
        if _should_skip_auto_extract(target_text):
            return chat_complete_default(system, user, temperature, max_tokens)

        # 1. Search for similar recipes
        similar_recipes = recipe_service.search_recipes(target_text, 2)
        print(f"🔍 Found {len(similar_recipes)} similar recipes in vector DB")
        print("Similar recipes:", similar_recipes)   
        if similar_recipes:
            # Format recipes for GPT context
            recipes_context = "\n\n".join(
                recipe
                for recipe in similar_recipes["documents"][0]
            )
            
            # Create selection prompt for GPT
            selection_prompt = f"""I found these similar recipes:

{recipes_context}

Based on the user's query: "{target_text}"

Please:
1. Choose the most relevant recipe
2. Always include complete ingredients with quantities
3. Ensure clear step-by-step instructions
4. If calorie information is missing, I will calculate it from ingredients

Your response should be well-structured with ingredients and steps."""

        # Let GPT choose/combine recipes
        request_kwargs = {
            "model": DEPLOYMENT,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": selection_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = client.chat.completions.create(**request_kwargs)
        recipe_response = response.choices[0].message.content.strip()

        resolved_ingredients: Optional[List[Dict[str, object]]] = None
        # Extract ingredients from the recipe response
        ingredients = extract_ingredients_from_prompt(recipe_response)
        
        if ingredients:
            print("🧾 Extracted ingredients from recipe:", json.dumps(ingredients, indent=2, ensure_ascii=False))
            _save_last_ingredients(ingredients)
            
            # Calculate calories if needed
            if _wants_calorie_total(target_text) or "calories" not in recipe_response.lower():
                result = calculate_calories(ingredients)
                calorie_info = _format_calorie_result(result)
                recipe_response += f"\n\nNutritional Information:\n{calorie_info}"
            
            return recipe_response
        
        # If no similar recipes found or extraction failed, fall back to default behavior
        print("⚠️ No similar recipes found, falling back to default completion")
        allow_tools = _wants_calorie_total(target_text)

        # 2️⃣ Nếu không có nguyên liệu → để GPT xử lý như bình thường
        request_kwargs = {
            "model": DEPLOYMENT,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if allow_tools:
            request_kwargs["tools"] = function_definition
            request_kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message

        # 3️⃣ Nếu GPT gọi function (tool_call)
        if getattr(message, "tool_calls", None) and allow_tools:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                if func_name == "calculate_calories" and allow_tools:
                    result = calculate_calories(args["ingredients"])
                    _save_last_ingredients(args["ingredients"])

                    follow_up = client.chat.completions.create(
                        model=DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                            message,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_name,
                                "content": json.dumps(result, ensure_ascii=False),
                            },
                        ],
                        tools=function_definition,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    final_reply = follow_up.choices[0].message.content.strip()
                    _remember_dish_ingredients(final_reply)
                    return final_reply

        # 4️⃣ Nếu GPT không gọi hàm — chỉ trả lời text
        final_content = (message.content or "").strip()
        detected = extract_ingredients_from_prompt(final_content)
        _remember_dish_ingredients(final_content, detected if detected else None)
        return final_content

    except Exception as e:
        if _is_transient_error(e):
            raise TransientOpenAIError(str(e))
        raise


# --------------------------
# Main summarization pipeline
# --------------------------
def summarize_transcript(text: str, cfg: Optional[SummarizeConfig] = None) -> str:
    cfg = cfg or SummarizeConfig()
    if cfg.style not in STYLE_GUIDES:
        raise ValueError(
            f"Unknown style '{cfg.style}'. Choose from: {list(STYLE_GUIDES)}"
        )

    style_desc = STYLE_GUIDES[cfg.style]
    chunks = chunk_text(text, MAX_CHARS_PER_CHUNK)

    partials: List[str] = []
    for idx, ch in enumerate(chunks, 3):
        # input is user: prompt of User
        user = USER_PROMPT.format(chunk=ch, style=style_desc, lang=cfg.lang)
        summary = chat_complete(SYSTEM_PROMPT, user, cfg.temperature, MAX_OUTPUT_TOKENS, raw_user_text=ch)
        partials.append(f"### Part {idx}\n{summary}")

    if len(partials) == 1:
        return partials[0].replace("### Part 1\n", "").strip()

    # the else case below usually not happen
    combined = "\n\n".join(partials)
    reducer_prompt = f"""Combine the following partial summaries into one cohesive final summary.
                        Keep the structure: Executive Summary, Key Decisions, Action Items, Risks/Blockers, Open Questions.
                        Write in language: {cfg.lang}. Style: {style_desc}. Avoid repetition.

                        Partial Summaries:
                        \"\"\"
                        {combined}
                        \"\"\"
                    """
    final_summary = chat_complete(SYSTEM_PROMPT, reducer_prompt, cfg.temperature, MAX_OUTPUT_TOKENS, raw_user_text="")
    return (final_summary or "").strip()
