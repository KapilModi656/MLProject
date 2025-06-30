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
    def detect_flat_matrix(text):
        """
        Detects space-separated numbers possibly forming a matrix
        and converts to LaTeX bmatrix if the pattern matches.
        """
        def matrix_repl(match):
            nums = match.group(0).strip().split()
            n = len(nums)
            if n < 4:
                return match.group(0)  # too small for matrix
            # try to guess number of columns (2, 3, 4, ...)
            for col in [2, 3, 4, 5, 6]:
                if n % col == 0:
                    rows = [' & '.join(nums[i:i+col]) for i in range(0, n, col)]
                    matrix = "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"
                    return f"$$\n{matrix}\n$$"
            return match.group(0)  # fallback

        return re.sub(r'(?<!\S)(-?\d+(\.\d+)?[\s,]+){3,}-?\d+(\.\d+)?(?!\S)', matrix_repl, text)

    def wrap_latex_blocks(text):
        # Wrap any raw \int, \frac, \sum, etc. in $$ if not already
        patterns = [
            r'\\int', r'\\sum', r'\\frac', r'\\lim', r'\\log', r'\\sqrt',
            r'\\left', r'\\right', r'\\begin\{bmatrix\}', r'\\end\{bmatrix\}',
            r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\^', r'_', r'\\cdot', r'\\times', r'dx', r'dy'
        ]
        combined = '|'.join(patterns)
        def wrap_math(match):
            expr = match.group(0)
            if not expr.startswith('$$'):
                return f"$$ {expr.strip()} $$"
            return expr
        return re.sub(rf'(?<!\$)\s*({combined}.*?)(?=\s|$)', wrap_math, text)

    def clean_custom_tags(text):
        return re.sub(r'</?think>', '', text)

    # Begin formatting
    text = clean_custom_tags(text)
    text = detect_flat_matrix(text)
    text = wrap_latex_blocks(text)

    # Escape stray square brackets if still left
    text = text.replace('[', '\\[').replace(']', '\\]')

    return text

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
        content = fix_latex_format(msg["content"])
        if msg["role"] == "user":
            st.markdown('', unsafe_allow_html=True)
            cols = st.columns([0.15, 0.85])
            with cols[1]:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(90deg, var(--user-bubble-bg, #388e3c) 80%, #43a047 100%); color: var(--user-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 100%; word-break: break-word; text-align: right; box-shadow: 0 2px 8px rgba(34,139,34,0.08); font-size: 1.08em;'>
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        else:
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(90deg, var(--assistant-bubble-bg, #23272f) 80%, #31363f 100%); color: var(--assistant-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 100%; word-break: break-word; text-align: left; box-shadow: 0 2px 8px rgba(0,0,0,0.10); font-size: 1.08em;'>
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# Improved CSS for dark/light mode compatibility and bubble style
st.markdown('''
    <style>
    html[data-theme="dark"] {
        --user-bubble-bg: #388e3c;
        --user-bubble-fg: #fff;
        --assistant-bubble-bg: #23272f;
        --assistant-bubble-fg: #fff;
        background-color: #18191a;
    }
    html[data-theme="light"] {
        --user-bubble-bg: #DCF8C6;
        --user-bubble-fg: #222;
        --assistant-bubble-bg: #F1F0F0;
        --assistant-bubble-fg: #222;
        background-color: #fff;
    }
    /* Make chat area a bit more modern */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
    }
    /* Hide Streamlit's default chat message background */
    [data-testid="stChatMessage"] {
        background: none !important;
        box-shadow: none !important;
    }
    </style>
''', unsafe_allow_html=True)

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
        st.markdown('', unsafe_allow_html=False)
        cols = st.columns([0.15, 0.85])
        with cols[1]:
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, var(--user-bubble-bg, #388e3c) 80%, #43a047 100%); color: var(--user-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 100%; word-break: break-word; text-align: right; box-shadow: 0 2px 8px rgba(34,139,34,0.08); font-size: 1.08em;'>
                    {fix_latex_format(user_text)}
                </div>
                """,
                unsafe_allow_html=True
            )

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
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(90deg, var(--assistant-bubble-bg, #23272f) 80%, #31363f 100%); color: var(--assistant-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 100%; word-break: break-word; text-align: left; box-shadow: 0 2px 8px rgba(0,0,0,0.10); font-size: 1.08em;'>
                        {fix_latex_format(assistant_msg)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

