import streamlit as st
from workflow import create_workflow
import re
graph = create_workflow()
# ----------------------------
# Latex Fixer
# ----------------------------
def fix_latex_format(text: str) -> str:
    # Replace [bmatrix] with {bmatrix}
    text = text.replace(r'\begin[bmatrix]', r'\begin{bmatrix}')
    text = text.replace(r'\end[bmatrix]', r'\end{bmatrix}')
    # Ensure double backslashes for row breaks
    text = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', text)
    # Optionally, wrap in $$ if not already
    if r'\begin{bmatrix}' in text and not text.strip().startswith('$$'):
        text = f"$$\n{text.strip()}\n$$"
    return text

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="MNITGPT", layout="centered")
st.title("🧠 MNITGPT")
st.markdown("This app uses a LangGraph workflow to intelligently process files, links, and prompts with LLMs.")

# --- API Key Management ---
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Enter your Groq API Key:",
        value=st.session_state["groq_api_key"],
        type="password",
        help="Your key is stored only in your browser session and never sent to anyone except Groq."
    )
    if groq_key != st.session_state["groq_api_key"]:
        st.session_state["groq_api_key"] = groq_key
        st.success("Groq API key updated!")

# Session Messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(fix_latex_format(msg["content"]), unsafe_allow_html=True)

# User Input
prompt = st.chat_input(placeholder="Enter your question or upload a file:", accept_file="multiple")

if prompt:
    user_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    user_files = prompt.files if hasattr(prompt, 'files') else []
    user_input = {"text": user_text, "files": user_files}

    # Show User Message
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(fix_latex_format(user_text), unsafe_allow_html=True)

    # Call Workflow
    with st.spinner("Processing your input..."):
        # Pass the Groq API key to the workflow if needed
        
        result = graph.invoke({"user_input": user_input,"groq_api_key": st.session_state["groq_api_key"]})
        response_text = result.get("response") if isinstance(result, dict) else None

    if response_text:
        assistant_msg = response_text.content if hasattr(response_text, "content") else str(response_text)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(fix_latex_format(assistant_msg), unsafe_allow_html=True)

            # Copy + Download options
   
        
