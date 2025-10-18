# Cookbook chatbot (Python, Azure OpenAI)

A chatbot that returns cooking guidelines and estimates calories from ingredient lists. Runs as a Streamlit web app.

---

## Features
- Streamlit web UI for user dish input and generated recipes.
- Auto-extracts ingredients (weights in grams) and computes calories using a local dataset.
- Handles long text via chunking & hierarchical summarization.
- Azure OpenAI integration (chat completions + optional function/tool calls).

---

## Prerequisites
- Python 3.9+
- Azure OpenAI resource with a chat-capable deployment.
- (Optional) dataset_with_vietnamese.json for calorie lookups.

---

## Environment Variables
Create a `.env` (or set in shell):
```
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_KEY="<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT="<your-deployment-name>"   # e.g., "gpt-4o-mini"
AZURE_OPENAI_API_VERSION="2024-07-01-preview"
```

Note: AZURE_OPENAI_DEPLOYMENT must be the exact deployment name you created in Azure.

---

## Install (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # or: env\Scripts\activate
pip install -r requirements.txt
```

---

## Run (Web UI)
```bash
streamlit run app_streamlit.py
```
Open the local URL printed by Streamlit and input a dish or paste a transcript.

---

## Troubleshooting

- Missing env vars → script raises EnvironmentError. Ensure `.env` or environment contains required keys.


- The repository contains an example change in `summarize.py` that collects all tool_call responses, creates tool messages for each, and issues one follow-up completion — this resolves the BadRequest.

- Rate limits / transient errors: code includes retries for transient OpenAI errors; wait and retry if you hit rate limits.

---

## Testing & Development
- Run tests:
```bash
pytest -q
```

---

## Data / Helpers
- dataset_with_vietnamese.json — lookup table for calories per 100g (keep in project root if used).
- Helper behavior:
  - Ingredients are normalized and converted to grams when possible.
  - If ingredient extraction fails, the app falls back to manual input or returns the normal chat completion.

---

## Example
How to text: Please use the text prompt below (Sample Prompts)
Hướng dẫn nấu món gà nấu nấm đông cô
 
Hướng dẫn nấu món gà nướng muối ớt
