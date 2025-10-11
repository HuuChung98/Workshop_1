
import argparse, sys, os
from dotenv import load_dotenv

from summarize import summarize_transcript, SummarizeConfig, STYLE_GUIDES

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Summarize a meeting transcript using Azure OpenAI.")
    parser.add_argument("--input", "-i", required=True, help="Path to input .txt transcript")
    parser.add_argument("--output", "-o", default="summary.txt", help="Path to write summary .txt")
    parser.add_argument("--style", "-s", default="executive", choices=list(STYLE_GUIDES.keys()))
    parser.add_argument("--lang", "-l", default="en", help="Language code (e.g., en, vi)")
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    cfg = SummarizeConfig(style=args.style, lang=args.lang, temperature=args.temperature)
    summary = summarize_transcript(text, cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Summary written to {args.output}")

if __name__ == "__main__":
    main()
