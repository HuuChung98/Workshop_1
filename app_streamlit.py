# app_streamlit_chat.py
import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from summarize import summarize_transcript, SummarizeConfig, STYLE_GUIDES

# ---------- Init ----------
load_dotenv()
st.set_page_config(page_title="Cooking Instructor (Chat)", page_icon="📝", layout="centered")

# ---------- Utilities ----------
def get_azure_status() -> Dict[str, str]:
    return {
        "Endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "(not set)"),
        "Deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "(not set)"),
        "API Version": os.getenv("AZURE_OPENAI_API_VERSION", "(not set)"),
    }

def ensure_session_state() -> None:
    """Initialize session_state keys for chat messages and config once."""
    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant"|"system", "content": str}
        st.session_state.messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "Bạn đang trò chuyện với Cooking Instructor. Hãy đặt câu hỏi về nấu ăn, công thức, tips & tricks.",
            }
        ]
    if "cfg" not in st.session_state:
        st.session_state.cfg = SummarizeConfig(
            style=list(STYLE_GUIDES.keys())[1] if len(STYLE_GUIDES) > 1 else list(STYLE_GUIDES.keys())[0],
            lang="en",
            temperature=0.3,
        )

def render_sidebar() -> None:
    st.sidebar.header("⚙️ Settings")
    # Summary options
    # style = st.sidebar.selectbox("Summary style", list(STYLE_GUIDES.keys()), index=1 if len(STYLE_GUIDES) > 1 else 0)
    # lang = st.sidebar.selectbox("Language", ["en", "vi"], index=0)
    temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 3.0, value=0.3, step=0.05)
    st.session_state.cfg = SummarizeConfig(temperature=temperature)

    # Azure status
    with st.sidebar.expander("🔐 Azure OpenAI Status", expanded=False):
        status = get_azure_status()
        for k, v in status.items():
            st.text(f"{k}: {v}")

    # Actions
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]  # keep system
        st.rerun()

    # Download conversation
    if len(st.session_state.messages) > 1:
        full_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages if m["role"] != "system"
        )
        st.sidebar.download_button(
            label="⬇️ Download transcript",
            data=full_text,
            file_name="conversation.txt",
            mime="text/plain",
        )

def render_effective_prompt() -> None:
    with st.expander("🧩 Effective Prompt (đang áp dụng)"):
        cfg: SummarizeConfig = st.session_state.cfg
        st.markdown(
            f"""
**Style**: `{cfg.style}`  
**Language**: `{cfg.lang}`  
**Temperature**: `{cfg.temperature}`

> *Gợi ý*: Bạn có thể yêu cầu:  
> - “Tóm tắt các ý chính & action items từ đoạn hội thoại sau …”  
> - “Đưa ra hướng dẫn nấu ăn cho người mới: nguyên liệu, định lượng, các bước, lưu ý an toàn …”
            """
        )

def assistant_reply(user_text: str) -> str:
    """
    Build a concise, helpful assistant reply using summarize_transcript() to keep a consistent tone/style.
    The user_text can be một prompt hỏi mới hoặc một đoạn hội thoại dài cần tóm tắt/hướng dẫn.
    """
    cfg: SummarizeConfig = st.session_state.cfg
    # Ở đây tận dụng summarize_transcript như một "brain" để chuẩn hóa văn phong/format
    # Lưu ý: summarize_transcript nên tự bảo toàn ngôn ngữ theo cfg.lang
    return summarize_transcript(user_text.strip(), cfg)

def render_chat_history() -> None:
    """Display chat messages from session_state."""
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------- App Body ----------
ensure_session_state()
st.title("📝 Cooking Instructor (Azure OpenAI · Chat Mode)")
st.caption("Người dùng hỏi → hiện prompt → trả lời → tiếp tục hỏi không gián đoạn.")

render_sidebar()
render_effective_prompt()
render_chat_history()

# Ô nhập chat mới
prompt = st.chat_input(
    placeholder=(
        "Gõ câu hỏi về nấu ăn / tóm tắt hội thoại / hướng dẫn chi tiết... "
        "Ví dụ: 'Hướng dẫn mình làm sốt cà chua kiểu Ý từ nguyên liệu sẵn có.'"
    )
)

if prompt:
    # 1) Show user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Assistant replies
    with st.chat_message("assistant"):
        with st.spinner("Đang soạn câu trả lời..."):
            try:
                reply = assistant_reply(prompt)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)
            except Exception as e:
                err = f"❌ Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.error(err)
