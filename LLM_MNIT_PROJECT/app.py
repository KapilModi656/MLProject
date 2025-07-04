import streamlit as st
from workflow import create_workflow
import re
import uuid
from utils import make_retriever,text_retriever,tut_retriever
import os
from datetime import timedelta
import time
graph = create_workflow()

syllabus_path=os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/syllabus"
tutorial_path= os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/tutorials"
pyq_path= os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/pyq"
docs_path= os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/docs/docs.txt"
@st.cache_resource(ttl=60*60*24*7)
def get_retrievers():
    retrievers = {
        "syllabus": make_retriever(syllabus_path),
        "tutorial": tut_retriever(tutorial_path),
        "pyq": make_retriever(pyq_path),
    }
    try:
        retrievers["docs"] = text_retriever(docs_path)
    except Exception as e:
        print("Error loading docs retriever:", e)
    return retrievers


retrievers = get_retrievers()


if "retriever_syllabus" not in st.session_state:
    st.session_state["retriever_syllabus"] = retrievers["syllabus"]

if "retriever_tutorial" not in st.session_state:
    st.session_state["retriever_tutorial"] = retrievers["tutorial"]

if "retriever_pyq" not in st.session_state:
    st.session_state["retriever_pyq"] = retrievers["pyq"]

if "file" not in st.session_state:
    st.session_state["file"] = []
if "text_retriever" not in st.session_state:
    st.session_state["text_retriever"] = retrievers["docs"]
if "processing" not in st.session_state:
    st.session_state["processing"] = False

def clean_custom_tags(text: str) -> str:
  
    return re.sub(r"</?think>", "", text)

def fix_latex_format(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    
    text = re.sub(r"</?think>", "", text)

    def wrap_in_dollars(match):
        content = match.group(1)
        return f"$$\n{content.strip()}\n$$"

 
    if not re.search(r"\$\$.*?\$\$", text, flags=re.DOTALL):
  
        text = re.sub(
            r'(?<!\$)\s*(\\begin\{bmatrix\}.*?\\end\{bmatrix\})\s*(?!\$)',
            wrap_in_dollars,
            text,
            flags=re.DOTALL
        )

    def detect_matrix_block(match):
        nums = match.group(0).strip().split()
        if len(nums) % 3 == 0 and all(n.replace('-', '').replace('.', '').isdigit() for n in nums):
            rows = [' & '.join(nums[i:i+3]) for i in range(0, len(nums), 3)]
            latex_matrix = "\\begin{bmatrix}\n" + ' \\\\\n'.join(rows) + "\n\\end{bmatrix}"
            return f"\n$$\n{latex_matrix}\n$$\n"
        return match.group(0)

   
    text = re.sub(r'(?<!\$)(?:\d+[\s\-]+)+\d+(?!\$)', detect_matrix_block, text)


    text = re.sub(r"\n{2,}", "\n\n", text.strip())

    return text.strip()

# Update the page configuration to include a favicon
st.set_page_config(
    page_title="MNITGPT",
    layout="centered",
    page_icon=os.getcwd() + "/img/mnitgpt.jpg"  # Path to the favicon image
)

# Replace the image-based logo with a styled text-based logo
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #388e3c;">
            🧠 MNITGPT
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    st.markdown("### 🔑 Groq API Key(Optional)")
    groq_key = st.text_input(
        "Enter Groq API Key(Optional):",
        value=st.session_state["groq_api_key"],
        type="password"
    )
    if groq_key != st.session_state["groq_api_key"]:
        st.session_state["groq_api_key"] = groq_key
        st.success("Updated!")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        content = msg["content"]
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


prompt = st.chat_input(
        placeholder="Ask about tutorials, research, or upload files...",
        accept_file="multiple",
    
    )

if prompt:
    
  
    user_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    user_files = prompt.files if hasattr(prompt, 'files') else []
    if user_files:
        st.session_state["file"] = user_files

    user_input = {"text": user_text, "files": st.session_state.get("file", [])}
   
    
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write('', unsafe_allow_html=False)
        cols = st.columns([0.15, 0.85])
        with cols[1]:
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, var(--user-bubble-bg, #388e3c) 80%, #43a047 100%); color: var(--user-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 100%; word-break: break-word; text-align: right; box-shadow: 0 2px 8px rgba(34,139,34,0.08); font-size: 1.08em;'>
                    {user_text}
                </div>
                """,
                unsafe_allow_html=True
            )

    
    with st.spinner("🧠 Thinking..."):
        
        result = graph.invoke({
            "user_input": user_input,
            "groq_api_key": st.session_state["groq_api_key"],
            "retriever_syllabus": st.session_state["retriever_syllabus"],
            "retriever_tutorial": st.session_state["retriever_tutorial"],
            "retriever_pyq": st.session_state["retriever_pyq"],
            "text_retriever": st.session_state["text_retriever"]
        })
        response_text = result.get("final_response") if isinstance(result, dict) else None
    
    if response_text:
     
      
        assistant_msg = fix_latex_format(response_text.content if hasattr(response_text, "content") else str(response_text))
        st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            cols = st.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(90deg, var(--assistant-bubble-bg, #23272f) 80%, #31363f 100%); color: var(--assistant-bubble-fg, #fff); padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 100%; word-break: break-word; text-align: left; box-shadow: 0 2px 8px rgba(0,0,0,0.10); font-size: 1.08em;'>
                        {assistant_msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.session_state["processing"] = False