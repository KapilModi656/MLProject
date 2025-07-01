from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from utils import (
    has_url, find_urls, is_file, docreader, doc_prompt_merger, type_url,
    youtube_reader, web_reader, wikipedia_tool, arxiv_tool, web_tool, save_uploaded_file, make_retreiver,
    theory_summarizer
)
import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from database import get_retriever

import re
load_dotenv()

# Initialize LLMs




class State(TypedDict):
    user_input: dict[str, str | list] | str
    response: str
    context: str
    urls: list[str]
    llm_choice: str
    docs: list[str]
    merged_prompt: str
    youtube_docs: list[str]
    web_docs: list[str]
    arxiv_docs: list[str]
    wiki_response: str
    web_response: str
    vectordb_docs: list[str]  # Assuming this is a list of documents from the vector database
    url_types: dict[str, list[str]]
    final_prompt: str
    data: str
    isfile: bool
    has_urls: bool
    groq_api_key: str
    syllabus: list[str]  # Assuming syllabus is a list of strings
    syllabus_choice: str
    tut_pyq: str
    tutorial: list[str]
    pyqs: list[str]
    retriever_syllabus:RunnableLambda
    retriever_tutorial:RunnableLambda
    retriever_pyq:RunnableLambda
    use_docs: bool


syllabus_path=os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/syllabus"
tutorial_path= os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/tutorials"
pyq_path= os.getcwd() + "/LLM_MNIT_PROJECT/1stSem/pyq"
def check_for_file_node(state: State):
    state["isfile"] = is_file(state["user_input"])
    print("check_for_file_node:working fine")
    return state

def FileConverter(state: State):
    state.setdefault("docs", [])
    files = state["user_input"].get("files", []) if isinstance(state["user_input"], dict) else []
    for file in files:
        path = save_uploaded_file(file)
        state["docs"] += docreader(path)
    print("FileConverter:working fine")
    return state

def prompt_doc_merger_node(state: State):
    text = state["user_input"].get("text", "")
    state["merged_prompt"] =doc_prompt_merger(text, state.get("docs", []) if state.get("docs") else text)
    print("prompt_doc_merger_node:working fine")

    return state

def check_for_urls_node(state: State):
    text = state["user_input"].get("text", "")
    urls = find_urls(text)
    state.setdefault("urls", []).extend(urls)
    state["has_urls"] = bool(urls)
    print("check_for_urls_node:working fine")
    return state

def url_type_node(state: State):
    state["url_types"] = type_url(state.get("urls", []))
    print("url_type_node:working fine")
    
    return state

def youtube_reader_node(state: State):
    state["youtube_docs"] = sum([youtube_reader(url) for url in state["url_types"].get("youtube", [])], [])
    print("youtube_reader_node:working fine")
    return state

def web_reader_node(state: State):
    print("web_reader_node:working fine")
    state["web_docs"] = sum([web_reader(url) for url in state["url_types"].get("web", [])], [])
    return state

def arxiv_reader_node(state: State):
    print("arxiv_reader_node:working fine")
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["arxiv_docs"] = arxiv_tool(text)
    return state

def wikipedia_reader_node(state: State):
    print("wikipedia_reader_node:working fine")
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["wiki_response"] = wikipedia_tool(text)
    return state

def web_reader_tool_node(state: State):
    print("web_reader_tool:working fine")
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["web_response"] = web_tool(text)
    return state

def final_prompt_node(state: State):
    print("final_prompt_node:working fine")
    docs_lists = [
        state.get("docs") or [],
        state.get("arxiv_docs") or [],
        state.get("youtube_docs") or [],
        state.get("web_docs") or [],
        state.get("vectordb_docs") or [],
        state.get("syllabus") or [],
        state.get("tutorial") or [],
        state.get("pyqs") or [],
    ]
    combined_docs = []
    for doc_list in docs_lists:
        combined_docs.extend(doc_list)

    context_parts = []
    if combined_docs:
        context_parts.append("\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in combined_docs
        ))
    for key in ["wiki_response", "web_response", "context"]:
        val = state.get(key)
        if val:
            context_parts.append(val)

    context = "\n\n".join(context_parts).strip()
    context=theory_summarizer(context=context)
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    final_prompt = r"""
    System: You are MNITGPT, an intelligent academic assistant developed by Kapil Modi for students, researchers, and professors at MNIT. Your primary role is to:

1. Provide simplified yet technically accurate explanations of academic topics, tutorial problems, and research papers using data from a curated vector database.
2. Detect whether the user is a student or researcher and adjust your tone, format, and depth accordingly.
3. Always format your output for Markdown compatibility. Use:
   - Bullet points, headers, and bold for structure.
   - LaTeX properly wrapped in `$$...$$` for display equations, and `$...$` for inline math.
   - Matrices in LaTeX using `\\begin{{bmatrix}} ... \\end{{bmatrix}}` and wrap them in `$$...$$`.
   - Integrals, derivatives, and summations using display-style LaTeX blocks (also in `$$...$$`).
   - Ensure all expressions render cleanly in Streamlit or Markdown viewers.
   - Use curly brackets or give response such that i can directly display it using markdown
4. Keep the response helpful, humble, and clearly structured — ranging from 2 to 400 words, depending on complexity.

5. When user uploads files or links (e.g., PDFs, papers, tutorials), extract and summarize core content or steps.

6. Never assume or hallucinate data beyond the provided context or the trusted vector database. If unsure, politely say so.
7. You would get summarized context from the theory_summarizer function and use it in your response.
8. dont give your intrusive thought in output only give direct answer to the user whether maths tutorial or pyq
Always maintain this identity and formatting standard.

---

Respond accordingly:

- 👩‍🎓 For Students → Use clear explanations, short step-by-step solutions, examples, visual cues (in ASCII/LaTeX), and optional references.
- 👨‍🏫 For PhD/Professors → Use formal tone, structured headings, assumptions, derivations, and references if available.

**Respond in Markdown-ready format.**
Wrap any LaTeX using `$$...$$` or `$...$` so it renders correctly in Streamlit or Markdown environments.

Context: {context}

User Input: {text}

"""
    state["final_prompt"] = final_prompt.format(
        context=context,
        text=text
    )

    print(f"[final_prompt_node] docs: {len(combined_docs)}, wiki_response: {bool(state.get('wiki_response'))}, web_response: {bool(state.get('web_response'))}, context: {bool(state.get('context'))}")
    print(f"[final_prompt_node] final_prompt set? {state['final_prompt'][:50]}")

    return state


def mnit_syllabus(state: State):
    """
    This node is a placeholder for syllabus-related processing.
    It can be extended to handle syllabus-specific logic.
    """
    # Example: Just echo the syllabus back
    retriever = state.get("retriever_syllabus")
    state["syllabus"] = retriever.invoke(state.get("user_input", {}).get("text", ""))
    return state
def tutorials(state: State):
    """
    This node is a placeholder for tutorials-related processing.
    It can be extended to handle tutorial-specific logic.
    """
    # Example: Just echo the tutorials back
    retriever = state.get("retriever_tutorial")
    state["tutorial"] = retriever.invoke(state.get("user_input", {}).get("text", ""))
    return state
def pyq(state: State):
    """
    This node is a placeholder for previous year questions (PYQ) related processing.
    It can be extended to handle PYQ-specific logic.
    """
    # Example: Just echo the PYQ back
    retriever = state.get("retriever_pyq")
    state["pyqs"] = retriever.invoke(state.get("user_input", {}).get("text", ""))
    return state
def vectordb_node(state: State):
    retriever = get_retriever()
    text = state["user_input"].get("text", "")

    result = retriever.invoke(text)
    print(f"[vectordb_node] context set? {bool(result)}")
    print(f"[vectordb_node] raw context: {result}")

    state["vectordb_docs"] = result
    return state

def llm1_node(state: State):
    llm1 = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=os.getenv("CEREBRAS_API_KEY"))
    prompt = state.get("merged_prompt") or state.get("final_prompt")
    if prompt is None:
        raise ValueError("Neither 'merged_prompt' nor 'final_prompt' is set in the state before calling llm1_node.")
    
    print(f"[llm1_node] Using prompt: {prompt[:50]}...")  # Log first 50 chars of prompt
    state["response"] = llm1.invoke(prompt)
    return state

def llm2_node(state: State):
    llm2 = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY"))
    prompt = state.get("merged_prompt") or state.get("final_prompt")
    if prompt is None:
        raise ValueError("Neither 'merged_prompt' nor 'final_prompt' is set in the state before calling llm2_node.")
    print(f"[llm2_node] Using prompt: {prompt[:50]}...")  # Log first 50 chars of prompt
    state["response"] = llm2.invoke(prompt)
    return state



router_llm = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=os.getenv("CEREBRAS_API_KEY"), temperature=0)
syllabus_prompt= PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the most appropriate context:
- "syllabus" for syllabus-related queries or academic queries
- "none" for other queries if its tutorial or pyq
Just reply with one word.[syllabus, none].
Prompt: {input}
""")
tutorial_pyq_prompt= PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the most appropriate context:
- "tutorial" for tutorials-related queries
- "pyq" for previous year questions related queries
- "none" for other queries
Just reply with one word.[tutorial, pyq, none].
Prompt: {input}
""")
routing_prompt = PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the most appropriate LLM:
- "groq" for fast and reasoning answers and high coding tasks
- "cerebras" for general queries

Just reply with one word: groq,or cerebras
Prompt: {input}
""")

data_routing_prompt = PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the appropriate context:
- wiki: for academic queries 
- websearch: for other normal queries
- arxiv: for academic papers and research
- none: if no specific context is needed
and remember if the topic is about tutorial or pyq send wiki
Just reply with one word.[wiki, websearch, arxiv,none].
Prompt: {input}
""")
use_docs_prompt= PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose whether it is related to document that uploaded or not:
- "yes" if the user is asking about the uploaded document or in a context of it or he says explain me this or input is blank or says explain me this pdf,ppt,docs etc
- "no" if the user is not asking about the uploaded document
                                            
give one word answer only.["yes", "no"].
docs:{context}
Prompt: {input}
                                              """)
def use_docs_node(state: State):
    text = state["user_input"].get("text", "")
    combined_docs=state.get("docs",[])
    context_parts = []
    if combined_docs:
        context_parts.append("\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in combined_docs
        ))
    summary_doc=theory_summarizer(context="\n\n".join(context_parts).strip())
    result = router_llm.invoke(use_docs_prompt.invoke({"input": text, "context": summary_doc}))
    use_docs_choice = result.content.strip().lower() if hasattr(result, "content") else str(result).lower()
    
    if use_docs_choice == "yes":
        state["use_docs"] = True
    elif use_docs_choice == "no":
        state["use_docs"] = False
    else:
        raise ValueError(f"Unexpected use_docs_choice: {use_docs_choice}")
    
    return state
def syllabus_node(state: State):
    text = state["user_input"].get("text", "")
    result = router_llm.invoke(syllabus_prompt.invoke({"input": text}))
    syllabus_choice = result.content.strip().lower() if hasattr(result, "content") else str(result).lower()
    state["syllabus_choice"] = syllabus_choice
    return state
def tutorials_pyq_node(state: State):
    text = state["user_input"].get("text", "")
    result = router_llm.invoke(tutorial_pyq_prompt.invoke({"input": text}))
    tut_pyq_choice = result.content.strip().lower() if hasattr(result, "content") else str(result).lower()
    
    if tut_pyq_choice == "tutorial":
        state["tut_pyq"] = "tutorial"
    elif tut_pyq_choice == "pyq":
        state["tut_pyq"] = "pyq"
    else:
        state["tut_pyq"] = "none"
    
    return state

def data_routing_node(state: State):
    text = state["user_input"].get("text", "")
    result = router_llm.invoke(data_routing_prompt.invoke({"input": text}))
    data_choice = result.content.strip().lower() if hasattr(result, "content") else str(result).lower()
    print(f"[data_routing_node] Selected context: {data_choice}")

    # Fallback to wiki if unexpected
    if data_choice not in ["wiki", "websearch", "arxiv", "none"]:
        data_choice = "wiki"
    
    state["data"] = data_choice
    
    # Optionally remove user_input if needed to avoid update errors
    # state = remove_user_input(state)  # if defined
    
    return state


def routing_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    result = router_llm.invoke(routing_prompt.invoke({"input": text})).content.strip().lower()
    # Remove punctuation and whitespace
    result = re.sub(r'[^a-z]', '', result)
    if result not in {"groq", "cerebras"}:
        raise ValueError(f"Unknown llm_choice: {result}")
    state["llm_choice"] = result
    print(f"[routing_node] Selected LLM: {state['llm_choice']}")
    return state


def create_workflow():
    graph = StateGraph(State)

    graph.add_node("check_for_file_node", check_for_file_node)
    graph.add_node("FileConverter", FileConverter)
    graph.add_node("prompt_doc_merger_node", prompt_doc_merger_node)
    graph.add_node("check_for_urls_node", check_for_urls_node)
    graph.add_node("url_type_node", url_type_node)
    graph.add_node("youtube_reader_node", youtube_reader_node)
    graph.add_node("web_reader_node", web_reader_node)
    graph.add_node("arxiv_reader_node", arxiv_reader_node)
    graph.add_node("wikipedia_reader_node", wikipedia_reader_node)
    graph.add_node("web_reader_tool_node", web_reader_tool_node)
    graph.add_node("data_routing_node", data_routing_node)
    graph.add_node("vectordb_node", vectordb_node)
    graph.add_node("final_prompt_node", final_prompt_node)
    graph.add_node("routing_node", routing_node)
    graph.add_node("syllabus_node", syllabus_node)
    graph.add_node("tutorials_pyq_node", tutorials_pyq_node)
    graph.add_node("mnit_syllabus", mnit_syllabus)
    graph.add_node("tutorials", tutorials)
    graph.add_node("pyq", pyq)
    graph.add_node("llm1_node", llm1_node)
    graph.add_node("llm2_node", llm2_node)
    graph.add_node("use_docs_node", use_docs_node)
    
    

    graph.add_edge(START, "check_for_file_node")
  
    graph.add_edge("FileConverter", "use_docs_node")
    graph.add_conditional_edges(
        "use_docs_node",
        lambda state: "yes" if state.get("use_docs", False) else "no",
        path_map={
            "yes": "prompt_doc_merger_node",
            "no": "check_for_urls_node"
        }
    )
    graph.add_edge("prompt_doc_merger_node", "routing_node")
    graph.add_edge("final_prompt_node", "routing_node")
    

    # LLM selection
    graph.add_conditional_edges(
        "routing_node",
        lambda state: state["llm_choice"],
        path_map={
            "groq": "llm2_node",
            "cerebras": "llm1_node",
            
        }
    )

    # LLM1/2/3 terminal
    graph.add_edge("llm1_node", END)
    graph.add_edge("llm2_node", END)
   

    # File check routing
    graph.add_conditional_edges(
        "check_for_file_node",
        lambda state: "True" if is_file(state["user_input"]) else "False",
        path_map={
            "True": "FileConverter",
            "False": "check_for_urls_node"
        }
    )

    graph.add_edge("prompt_doc_merger_node", "routing_node")

    # URL presence
    graph.add_conditional_edges(
        "check_for_urls_node",
        lambda state: "has_urls" if state.get("urls") else "no_urls",
        path_map={
            "has_urls": "url_type_node",
            "no_urls": "syllabus_node"
        }
    )
    graph.add_conditional_edges(
        "syllabus_node",
        lambda state: state.get("syllabus_choice","none"),
        path_map={
            "syllabus": "mnit_syllabus",
            "none": "tutorials_pyq_node"
        }
    )
    graph.add_edge("mnit_syllabus", "tutorials_pyq_node")
    graph.add_conditional_edges(
        "tutorials_pyq_node",
        lambda state: state.get("tut_pyq","None"),
        path_map={
            "tutorial": "tutorials",
            "pyq": "pyq",
            "none": "data_routing_node"
        }
    )
    graph.add_edge("tutorials", "data_routing_node")
    graph.add_edge("pyq", "data_routing_node")
    

    # URL types
    graph.add_conditional_edges(
        "url_type_node",
        lambda state: "youtube" if state["url_types"].get("youtube") else (
                      "arxiv" if state["url_types"].get("arxiv") else "web"),
        path_map={
            "youtube": "youtube_reader_node",
            "arxiv": "arxiv_reader_node",
            "web": "web_reader_node"
        }
    )

    graph.add_conditional_edges(
        "youtube_reader_node",
        lambda state: "has_youtube_docs" if state.get("youtube_docs") else "no_youtube_docs",
        path_map={
            "has_youtube_docs": "final_prompt_node",
            "no_youtube_docs": "arxiv_reader_node"
        }
    )
    graph.add_conditional_edges(
        "web_reader_node",
        lambda state: "has_web_docs" if state.get("web_docs") else "no_web_docs",
        path_map={
            "has_web_docs": "final_prompt_node",
            "no_web_docs": "arxiv_reader_node"
        }
    )
    graph.add_conditional_edges(
        "arxiv_reader_node",
        lambda state: "has_arxiv_docs" if state.get("arxiv_docs") else "no_arxiv_docs",
        path_map={
            "has_arxiv_docs": "final_prompt_node",
            "no_arxiv_docs": "wikipedia_reader_node"
        }
    )
    graph.add_conditional_edges(
        "wikipedia_reader_node",
        lambda state: "has_wiki_response" if state.get("wiki_response") else "no_wiki_response",
        path_map={
            "has_wiki_response": "final_prompt_node",
            "no_wiki_response": "web_reader_tool_node"
        }
    )
    graph.add_conditional_edges(
        "web_reader_tool_node",
        lambda state: "has_web_response" if state.get("web_response") else "no_web_response",
        path_map={
            "has_web_response": "final_prompt_node",
            "no_web_response": "final_prompt_node"
        }
    )
    graph.add_conditional_edges(
        "data_routing_node",
        lambda state: state.get("data", "wiki"),  # default to wiki if missing
        path_map={
            "wiki": "wikipedia_reader_node",
            "websearch": "web_reader_tool_node",
            "arxiv": "arxiv_reader_node",
            "none":"final_prompt_node"  # No context needed, go directly to final prompt
        }
    )
    graph.add_conditional_edges(
        "vectordb_node",
        lambda state: "has_context" if state.get("context") else "no_context",
        path_map={
            "has_context": "final_prompt_node",
            "no_context": "final_prompt_node"
        }
    )
    graph.add_conditional_edges(
        "final_prompt_node",
        lambda state: "has_docs" if state.get("docs") else "no_docs",
        path_map={
            "has_docs": "routing_node",
            "no_docs": "routing_node"
        }
    )

    return graph.compile()

graph= create_workflow()
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())  # Save the graph as a PNG file

