from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from utils import (
    has_url, find_urls, is_file, docreader, doc_prompt_merger, type_url,
    youtube_reader, web_reader, wikipedia_tool, arxiv_tool, web_tool, save_uploaded_file
)
import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from vector import get_retriever
from IPython.display import Image, display
load_dotenv()

# Initialize LLMs
llm1 = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=os.getenv("CEREBRAS_API_KEY"))
llm2 = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))
llm3 = ChatMistralAI(model="open-codestral-mamba", api_key=os.getenv("MISTRAL_API_KEY"))

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
    url_types: dict[str, list[str]]
    final_prompt: str
    data: str
    isfile: bool
    has_urls: bool

def remove_user_input(state: State):
    # Helper: copy state and remove user_input key to avoid concurrent updates error
    branch_state = state.copy()
    branch_state.pop("user_input", None)
    return branch_state

def check_for_file_node(state: State):
    state["isfile"] = is_file(state["user_input"])
    branch_state = remove_user_input(state)
    return {"True": branch_state} if state["isfile"] else {"False": branch_state}

def FileConverter(state: State):
    state.setdefault("docs", [])
    files = state["user_input"].get("files", []) if isinstance(state["user_input"], dict) else []
    for file in files:
        path = save_uploaded_file(file)
        state["docs"] += docreader(path)
    return state

def prompt_doc_merger_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["merged_prompt"] = doc_prompt_merger(text, state.get("docs", [])) if state.get("docs") else text
    return state

def check_for_urls_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    urls = find_urls(text)
    state.setdefault("urls", []).extend(urls)
    state["has_urls"] = bool(urls)
    branch_state = remove_user_input(state)
    return {"has_urls": branch_state} if urls else {"no_urls": branch_state}

def url_type_node(state: State):
    state["url_types"] = type_url(state.get("urls", []))
    branch_state = remove_user_input(state)
    if state["url_types"].get("youtube"):
        return {"youtube": branch_state}
    elif state["url_types"].get("arxiv"):
        return {"arxiv": branch_state}
    return {"web": branch_state}

def youtube_reader_node(state: State):
    state["youtube_docs"] = sum([youtube_reader(url) for url in state["url_types"].get("youtube", [])], [])
    branch_state = remove_user_input(state)
    return {"has_youtube_docs": branch_state} if state["youtube_docs"] else {"no_youtube_docs": branch_state}

def web_reader_node(state: State):
    state["web_docs"] = sum([web_reader(url) for url in state["url_types"].get("web", [])], [])
    branch_state = remove_user_input(state)
    return {"has_web_docs": branch_state} if state["web_docs"] else {"no_web_docs": branch_state}

def arxiv_reader_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["arxiv_docs"] = arxiv_tool(text)
    branch_state = remove_user_input(state)
    return {"has_arxiv_docs": branch_state} if state["arxiv_docs"] else {"no_arxiv_docs": branch_state}

def wikipedia_reader_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["wiki_response"] = wikipedia_tool(text)
    branch_state = remove_user_input(state)
    return {"has_wiki_response": branch_state} if state["wiki_response"] else {"no_wiki_response": branch_state}

def web_reader_tool_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["web_response"] = web_tool(text)
    branch_state = remove_user_input(state)
    return {"has_web_response": branch_state} if state["web_response"] else {"no_web_response": branch_state}

def final_prompt_node(state: State):
    docs_lists = [
        state.get("docs") or [],
        state.get("arxiv_docs") or [],
        state.get("youtube_docs") or [],
        state.get("web_docs") or []
    ]
    combined_docs = []
    for doc_list in docs_lists:
        combined_docs.extend(doc_list)

    context_parts = []
    if combined_docs:
        context_parts.append("\n".join(combined_docs))
    for key in ["wiki_response", "web_response", "context"]:
        val = state.get(key)
        if val:
            context_parts.append(val)

    context = "\n\n".join(context_parts).strip()
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["final_prompt"] = f"Context: {context}\nUser Input: {text}"

    print(f"[final_prompt_node] docs: {len(combined_docs)}, wiki_response: {bool(state.get('wiki_response'))}, web_response: {bool(state.get('web_response'))}, context: {bool(state.get('context'))}")
    print(f"[final_prompt_node] final_prompt set? {state['final_prompt'][:50]}")

    branch_state = remove_user_input(state)
    if context:
        return {"has_docs": branch_state}
    else:
        return {"no_docs": branch_state}

def ensure_prompt_node(state: State):
    if not state.get("merged_prompt") and not state.get("final_prompt"):
        text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
        state["final_prompt"] = f"User Input: {text}"
        print("[ensure_prompt_node] final_prompt forcibly set.")
    return state

def vectordb_node(state: State):
    retriever = get_retriever()
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["context"] = retriever.invoke(text)
    branch_state = remove_user_input(state)
    return {"has_context": branch_state} if state["context"] else {"no_context": branch_state}

def llm1_node(state: State):
    prompt = state.get("merged_prompt") or state.get("final_prompt")
    if prompt is None:
        raise ValueError("Neither 'merged_prompt' nor 'final_prompt' is set in the state before calling llm1_node.")
    state["response"] = llm1.invoke(prompt)
    return state

def llm2_node(state: State):
    prompt = state.get("merged_prompt") or state.get("final_prompt")
    if prompt is None:
        raise ValueError("Neither 'merged_prompt' nor 'final_prompt' is set in the state before calling llm2_node.")
    state["response"] = llm2.invoke(prompt)
    return state

def llm3_node(state: State):
    prompt = state.get("merged_prompt") or state.get("final_prompt")
    if prompt is None:
        raise ValueError("Neither 'merged_prompt' nor 'final_prompt' is set in the state before calling llm3_node.")
    state["response"] = llm3.invoke(prompt)
    return state

router_llm = ChatCerebras(model="llama-4-scout-17b-16e-instruct", api_key=os.getenv("CEREBRAS_API_KEY"), temperature=0)

routing_prompt = PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the most appropriate LLM:
- "groq" for fast and reasoning answers
- "cerebras" for general queries
- "mistral" for coding tasks
Just reply with one word: groq, cerebras, or mistral.
Prompt: {input}
""")

data_routing_prompt = PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the appropriate context:
- vectordb
- websearch
- arxiv
Just reply with one word.
Prompt: {input}
""")

def data_routing_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    result = router_llm.invoke(data_routing_prompt.invoke({"input": text}))
    data_choice = result.content.strip().lower() if hasattr(result, "content") else str(result).lower()
    
    # Fallback to vectordb if unexpected
    if data_choice not in ["vectordb", "websearch", "arxiv"]:
        data_choice = "vectordb"
    
    state["data"] = data_choice
    
    # Optionally remove user_input if needed to avoid update errors
    # state = remove_user_input(state)  # if defined
    
    return {data_choice: state}


def routing_node(state: State):
    text = state["user_input"].get("text", "") if isinstance(state["user_input"], dict) else state["user_input"]
    state["llm_choice"] = router_llm.invoke(routing_prompt.invoke({"input": text})).content.strip().lower()
    branch_state = remove_user_input(state)
    return branch_state

def llm_choice_node(state: State):
    llm = state.get("llm_choice", "cerebras")  # default to cerebras
    branch_state = remove_user_input(state)
    return {llm: branch_state} if llm in ["groq", "cerebras", "mistral"] else {"cerebras": branch_state}

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
    graph.add_node("llm_choice_node", llm_choice_node)
    graph.add_node("llm1_node", llm1_node)
    graph.add_node("llm2_node", llm2_node)
    graph.add_node("llm3_node", llm3_node)
    graph.add_node("ensure_prompt_node", ensure_prompt_node)

    graph.add_edge(START, "check_for_file_node")
    graph.add_edge("routing_node", "llm_choice_node")
    graph.add_edge("prompt_doc_merger_node", "ensure_prompt_node")
    graph.add_edge("final_prompt_node", "ensure_prompt_node")
    graph.add_edge("ensure_prompt_node", "routing_node")

    # LLM selection
    graph.add_conditional_edges(
        "llm_choice_node",
        lambda state: state["llm_choice"],
        path_map={
            "groq": "llm2_node",
            "cerebras": "llm1_node",
            "mistral": "llm3_node"
        }
    )

    # LLM1/2/3 terminal
    graph.add_edge("llm1_node", END)
    graph.add_edge("llm2_node", END)
    graph.add_edge("llm3_node", END)

    # File check routing
    graph.add_conditional_edges(
        "check_for_file_node",
        lambda state: "True" if is_file(state["user_input"]) else "False",
        path_map={
            "True": "FileConverter",
            "False": "check_for_urls_node"
        }
    )
    graph.add_edge("FileConverter", "prompt_doc_merger_node")
    graph.add_edge("prompt_doc_merger_node", "routing_node")

    # URL presence
    graph.add_conditional_edges(
        "check_for_urls_node",
        lambda state: "has_urls" if state.get("urls") else "no_urls",
        path_map={
            "has_urls": "url_type_node",
            "no_urls": "data_routing_node"
        }
    )

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
        lambda state: state.get("data", "vectordb"),  # default to vectordb if missing
        path_map={
            "vectordb": "vectordb_node",
            "websearch": "web_reader_tool_node",
            "arxiv": "arxiv_reader_node"
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

graph = create_workflow()
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
