import streamlit as st
from workflow import create_workflow  # make sure this file is error-free and in the same dir

# Initialize LangGraph





# Streamlit UI
st.set_page_config(page_title="MNITGPT", layout="centered")
st.title("🧠 MNITGPT")
st.markdown("This app uses a LangGraph workflow to intelligently process files, links, and prompts with LLMs.")

# File upload


# Text input
prompt = st.chat_input(placeholder="Or enter your query here:",accept_file="multiple")

# Combine inputs
if prompt:
    user_input = {"text": prompt.text, "files": prompt.files} if hasattr(prompt, 'files') else {"text": prompt, "files": []}
    if "message" not in st.session_state:
        text = user_input["text"] if isinstance(user_input["text"], str) else str(user_input["text"])
        st.session_state["message"] = text
        with st.chat_message(name ="user"):
            st.write(text)
    else:
        text = user_input["text"] if isinstance(user_input["text"], str) else str(user_input["text"])
        st.session_state["message"] += f"\n{text}"
    with st.spinner("Processing your input..."):
        

        
        state = {"user_input": user_input}
        
        graph = create_workflow()
        # Initial state to send to LangGraph
        

        # Call the LangGraph
        result = graph.invoke(state)
        response_text = result.get("response") if isinstance(result, dict) else None
        # Display result
        with st.chat_message("assistant"):
            st.write(response_text.content)

        st.session_state["message"] += f"\n{response_text.content}"
        
