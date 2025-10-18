Author: CHUNGLH3
DATE: 08/10/2025

# Meeting Summarizer (Python, Azure OpenAI)

A minimal app that lets users upload meeting transcripts and returns concise summaries.
You can run it as a **web app (Streamlit)** or a **CLI tool**.

---

## Features (Expected Outcomes)
- **Functional Application:** Upload a `.txt` transcript and receive a summarized version.
- **Simple UI:** Streamlit web UI with file upload + output panel; CLI for terminal use.
- **Handles long inputs:** Automatic chunking & hierarchical summarization (map-reduce).
- **Azure OpenAI Integration:** Uses `AzureOpenAI` SDK and `gpt-4o-mini`/compatible deployments.
- **Clear Docs:** This README plus inline code comments.

---

## Architecture (Concepts Covered)
- **GPT summarization** via Azure OpenAI Chat Completions.
- **Large text processing** with safe chunking, token-aware batching, and a map-reduce strategy.
- **UI design** with Streamlit: one page, one upload, one result.
- **Azure OpenAI integration** via environment variables.
- **Documentation** for setup, run, and customization.

---

## ⚙️ Prerequisites
- Python 3.9+
- An **Azure OpenAI** resource with a deployed chat model (`gpt-4o-mini` compatible).
- Your Azure OpenAI credentials and endpoint.

---

## 🔐 Environment Variables
Create `.env` (or set in your shell) with:
```
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_KEY="<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT="<your-deployment-name>"   # e.g., "gpt-4o-mini"
AZURE_OPENAI_API_VERSION="2024-07-01-preview"
```

> **Note:** The `AZURE_OPENAI_DEPLOYMENT` is the **deployment name** you created in Azure for the chosen model (not the model family).

---

## 📦 Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Run (Web UI)
```bash
streamlit run app_streamlit.py
```
Open the printed local URL in your browser. Upload a `.txt` transcript and click **Summarize**.

---

## 🧰 Run (CLI)
```bash
python summarize_cli.py --input sample_transcript.txt --output summary.txt
```
- Use `--style` to change tone (`executive`, `bullet`, `action-items`, `detailed`).  
- Use `--lang` to get summaries in another language (e.g., `vi` or `en`).

---

## 🧪 Quick Test
A small example is included at `sample_transcript.txt`:
```bash
python summarize_cli.py --input sample_transcript.txt --output summary.txt --style bullet
```


How to text: Please use the text prompt below (Sample Prompts)
Hướng dẫn nấu món gà nấu nấm đông cô
 
Hướng dẫn nấu món gà nướng muối ớt