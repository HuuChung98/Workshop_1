import os
from dataclasses import dataclass
from typing import List, Dict, Optional

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
                
                    - 1 con gà (khoảng 1.5 kg)
                    - 2 muỗng canh muối
                    - 1 muỗng canh tiêu
                    - 1 muỗng canh ớt bột
                    - 2 muỗng canh dầu ăn

                Instructions step by step(at least 5 steps and no more than 10 steps):
                
                    - 1: Quay gà sạch và để ráo nước.
                    - 2: Trộn đều muối, tiêu, ớt bột và dầu ăn trong một bát nhỏ.
                    - 3: Xoa hỗn hợp gia vị lên bề mặt gà, đảm bảo thấm đều.
                    - 4: Để gà ướp trong 30 phút cho ngấm gia vị.
                    - 5: Nướng gà trong lò đã được làm nóng trước ở 200 độ C trong 1 giờ.
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


function_definition = [
    {
        "type": "function",
        "function": {
            "name": "calculate_calories",
            "description": (
                "This function calculates the total calories based on the ingredient name and weight in grams."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient_name": {
                        "type": "string",
                        "description": (
                            "The name of the ingredient to calculate calories for."
                        ),
                    },
                    "weight": {
                        "type": "number",
                        "description": ("The weight of the ingredient in grams."),
                    },
                },
            },
            "result": {"type": "string"},
        },
    }
]

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


def calculate_calories(ingredients: list) -> dict:
    """
    Tính tổng lượng calories dựa trên danh sách nguyên liệu.
    Mỗi phần tử trong list có dạng:
    {"ingredient_name": "Cá hồi", "weight": 1000}
    """
    results = []
    total_calories = 0.0

    for item in ingredients:
        name = item["ingredient_name"]
        weight = item["weight"]
        # Tìm nguyên liệu trong dataset
        match = next(
            (d for d in dataset if d["Ingredients"].lower() == name.lower()), None
        )

        if not match:
            results.append(
                {
                    "ingredient": name,
                    "weight": weight,
                    "calories": None,
                    "note": "Không tìm thấy nguyên liệu trong dataset",
                }
            )
            continue

        cal_per_100g = match["Calories per 100g"]
        calories = (weight / 100) * cal_per_100g
        total_calories += calories

        results.append(
            {"ingredient": name, "weight": weight, "calories": round(calories, 2)}
        )

    return {"details": results, "total_calories": round(total_calories, 2)}


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


def extract_ingredients_from_prompt(prompt: str):
    """
    Trích xuất danh sách nguyên liệu và trọng lượng (gram) từ chuỗi người dùng nhập.
    Ví dụ: "Cá hồi: 1000g, Gạo trắng: 200g"
    """
    text = prompt.strip()
    pattern = r"([\w\sÀ-ỹ]+?)[:：]\s*(\d+(?:[\.,]\d+)?)\s*(?:g|gr|gram|grams)\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if len(matches) == 0:
        return []

    ingredients = []
    for name, weight_str in matches:
        try:
            weight = float(weight_str.replace(",", "."))
            ingredients.append(
                {"ingredient_name": name.strip().capitalize(), "weight": weight}
            )
        except ValueError:
            continue

    return ingredients


# --------------------------
# Chat completion with assistant role
# --------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def chat_complete(
    system: str, user: str, temperature: float, max_tokens: int = MAX_OUTPUT_TOKENS
) -> str:
    """
    Calls Azure OpenAI ChatCompletion with retry on transient errors.
    Tự động:
    - Bóc tách nguyên liệu nếu có ("Cá hồi: 100g, Gạo: 50g") và tính calories ngay.
    - Nếu không có nguyên liệu, để GPT xử lý theo flow thông thường.
    """
    try:
        # 1️⃣ Thử trích xuất danh sách nguyên liệu từ prompt
        ingredients = extract_ingredients_from_prompt(user)
        if ingredients and len(ingredients) > 0:
            print(
                "🧾 Ingredients auto-detected:",
                json.dumps(ingredients, indent=2, ensure_ascii=False),
            )
            result = calculate_calories(ingredients)

            # Format kết quả trả về
            lines = [f"Tổng calories: {result['total_calories']} kcal"]
            for item in result["details"]:
                if item["calories"]:
                    lines.append(
                        f"- {item['ingredient']} ({item['weight']}g): {item['calories']} kcal"
                    )
                else:
                    lines.append(
                        f"- {item['ingredient']} ({item['weight']}g): không có dữ liệu"
                    )

            return "\n".join(lines)

        # 2️⃣ Nếu không có nguyên liệu → để GPT xử lý như bình thường
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=function_definition,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message

        # 3️⃣ Nếu GPT gọi function (tool_call)
        if getattr(message, "tool_calls", None):
            # Build tool response messages for ALL tool_call_ids returned by the assistant.
            # Azure requires each assistant tool_call to be followed by a corresponding tool message.
            tool_messages = []
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                if func_name == "calculate_calories":
                    result = calculate_calories([args])
                    content = json.dumps(result, ensure_ascii=False)
                else:
                    # Provide a generic response for unsupported tool calls so every tool_call_id is answered.
                    content = json.dumps(
                        {"error": "unsupported tool", "tool": func_name},
                        ensure_ascii=False,
                    )

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": content,
                    }
                )

            # Send a single follow-up including all tool responses
            follow_up = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    message,
                    *tool_messages,
                ],
                tools=function_definition,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return follow_up.choices[0].message.content.strip()

        # 4️⃣ Nếu GPT không gọi hàm — chỉ trả lời text
        return (message.content or "").strip()

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
        summary = chat_complete(SYSTEM_PROMPT, user, cfg.temperature, MAX_OUTPUT_TOKENS)
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
    final_summary = chat_complete(
        SYSTEM_PROMPT, reducer_prompt, cfg.temperature, MAX_OUTPUT_TOKENS
    )
    return (final_summary or "").strip()
