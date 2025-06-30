import streamlit as st
from workflow import create_workflow
import re
import uuid

graph = create_workflow()

# ----------------------------
# Utilities
# ----------------------------
def clean_custom_tags(text: str) -> str:
    # Remove <think>...</think> and similar tags
    return re.sub(r"</?think>", "", text)

def fix_latex_format(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = clean_custom_tags(text)

    # Protect LaTeX blocks so we don't escape inside them
    def protect_latex_blocks(text):
        latex_blocks = []
        def repl(match):
            latex_blocks.append(match.group(0))
            return f"__LATEX_BLOCK_{len(latex_blocks)-1}__"
        return re.sub(
            r'(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{.*?\}.*?\\end\{.*?\})',
            repl, text, flags=re.DOTALL
        ), latex_blocks

    def restore_latex_blocks(text, latex_blocks):
        for i, block in enumerate(latex_blocks):
            text = text.replace(f"__LATEX_BLOCK_{i}__", block)
        return text

    text, latex_blocks = protect_latex_blocks(text)
    text = text.replace('[', '\\[').replace(']', '\\]')
    return restore_latex_blocks(text, latex_blocks)

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="MNITGPT", layout="centered")
st.title("🧠 MNITGPT")
st.markdown("This app helps MNIT students and researchers with tutorial solutions and research paper explanations.")

# ----------------------------
# API Key (Optional)
# ----------------------------
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Enter Groq API Key:",
        value=st.session_state["groq_api_key"],
        type="password"
    )
    if groq_key != st.session_state["groq_api_key"]:
        st.session_state["groq_api_key"] = groq_key
        st.success("Updated!")

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ----------------------------
# Show Chat History
# ----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        msg_id = str(uuid.uuid4()).replace('-', '')
        st.markdown(fix_latex_format(msg["content"]), unsafe_allow_html=True)
        if msg["role"] == "assistant":
            with st.expander("📋 Copy", expanded=False):
                st.code(msg["content"], language="markdown")

# ----------------------------
# Chat Input
# ----------------------------
prompt = st.chat_input(placeholder="Ask about tutorials, research, or upload files...", accept_file="multiple")

if prompt:
    user_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    user_files = prompt.files if hasattr(prompt, 'files') else []
    user_input = {"text": user_text, "files": user_files}

    # Show User Message
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(fix_latex_format(user_text), unsafe_allow_html=False)

    # Call LangGraph
    with st.spinner("🧠 Thinking..."):
        result = graph.invoke({
            "user_input": user_input,
            "groq_api_key": st.session_state["groq_api_key"]
        })
        response_text = result.get("response") if isinstance(result, dict) else None

    if response_text:
        assistant_msg = response_text.content if hasattr(response_text, "content") else str(response_text)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(fix_latex_format(assistant_msg), unsafe_allow_html=True)
            with st.expander("📋 Copy", expanded=False):
                st.code(assistant_msg, language="markdown")
