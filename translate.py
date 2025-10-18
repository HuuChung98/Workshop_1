import json
from deep_translator import GoogleTranslator

# --- Load dataset gốc ---
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

translator = GoogleTranslator(source="en", target="vi")
new_data = []

for item in data:
    new_data.append(item)
    vi_item = item.copy()
    try:
        vi_item["Ingredients"] = translator.translate(item["Ingredients"])
    except Exception as e:
        vi_item["Ingredients"] = item["Ingredients"]  # fallback nếu lỗi
    new_data.append(vi_item)

# --- Xuất file mới ---
with open("dataset_with_vietnamese.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Đã tạo file dataset_with_vietnamese.json với bản dịch tiếng Việt.")
