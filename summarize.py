import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AzureOpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

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
You are a cooking instructor. Your goal is to guide users through cooking processes by providing clear, structured, and easy-to-follow cooking instructions.

If the user requests more than 100 dishes, please respond:
    “Sorry, I can only provide up to 100 dishes.”

If the user ask outside cooking context, please respond:
    “Sorry, can not support outside cooking context.”

If user ask by Vietnamese, please respond in Vietnamese. If user ask by English, please respond in English.

Ensure that all generated answers maintain semantic equivalence either in any language given, preserving identical content, structure, quantities, steps, and warnings for complete functional and semantic consistency.

When suggesting dishes, always respond using the following format:
For the dish [summarize and interpret the user’s context], you may consider cooking the following: (suggest 3 dishes if the user has not specified any)
    Dish Name:
    Flavor Profile:

When providing cooking instructions, follow this structure:
    Dish Name
    Ingredients:
      ...
	    ...
	    ...

		Instructions (at least 5 steps and no more 10 steps):
	    Step 1: ...
	    Step 2: ...
	    Step 3: ...
    
		Caution on allergy if any
		
User question as below:
"""

USER_PROMPT = """
For the dish [summarize and interpret the user’s context], you may consider cooking the following 
(suggest 3 dishes if the user has not specified any):

    Dish Name:
    Flavor Profile:

When providing cooking instructions, follow this structure:

    Dish Name
    Ingredients:
      - ...
      - ...
      - ...

    Instructions (at least 5 steps and no more than 10 steps):
      Step 1: ...
      Step 2: ...
      Step 3: ...
      Step 4: ...
      Step 5: ...
      (Add more steps as needed, up to Step 10)

    Caution on allergy if any

Now, please analyze the following user input or transcript carefully and generate the corresponding cooking guide:

\"\"\"
{chunk}
\"\"\"



If the input is unclear or missing essential information (e.g., cuisine type, main ingredient, dish type, meal time), 
politely ask the user up to three short clarification questions before proceeding.
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


# --------------------------
# Chat completion with assistant role
# --------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def chat_complete(system: str, user: str, temperature: float, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """
    Calls Azure OpenAI ChatCompletion with retry on transient errors.
    Includes assistant role for context continuity.
    """
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": "Understood. Summarizing the meeting as requested..."},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content
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
        raise ValueError(f"Unknown style '{cfg.style}'. Choose from: {list(STYLE_GUIDES)}")

    style_desc = STYLE_GUIDES[cfg.style]
    chunks = chunk_text(text, MAX_CHARS_PER_CHUNK)

    partials: List[str] = []
    for idx, ch in enumerate(chunks, 3):
        user = USER_PROMPT.format(chunk=ch, style=style_desc, lang=cfg.lang)
        summary = chat_complete(SYSTEM_PROMPT, user, cfg.temperature, MAX_OUTPUT_TOKENS)
        partials.append(f"### Part {idx}\n{summary}")

    if len(partials) == 1:
        return partials[0].replace("### Part 1\n", "").strip()

    combined = "\n\n".join(partials)
    reducer_prompt = f"""Combine the following partial summaries into one cohesive final summary.
Keep the structure: Executive Summary, Key Decisions, Action Items, Risks/Blockers, Open Questions.
Write in language: {cfg.lang}. Style: {style_desc}. Avoid repetition.

Partial Summaries:
\"\"\"
{combined}
\"\"\"
"""
    final_summary = chat_complete(SYSTEM_PROMPT, reducer_prompt, cfg.temperature, MAX_OUTPUT_TOKENS)
    return (final_summary or "").strip()
