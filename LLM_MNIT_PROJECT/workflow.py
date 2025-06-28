from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from utils import has_url,find_urls,is_file,docreader,doc_prompt_merger,type_url,youtube_reader,web_reader,wikipedia_tool,arxiv_tool,web_tool,save_uploaded_file
import tempfile
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_mistral import ChatMistral
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
load_dotenv()
os.environ["CEREBRAS_API_KEY"] = os.getenv("CEREBRAS_API_KEY")
os.environ["LangChain_API_KEY"] = os.getenv("LangChain_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

llm1 = ChatCerebras(model="llama-4-scout-17b-16e-instruct",api_key=os.getenv("CEREBRAS_API_KEY"))
llm2 = ChatGroq(model="deepseek-r1-distill-llama-70b",api_key=os.getenv("GROQ_API_KEY"))
llm3 = ChatMistral(model="open-codestral-mamba",api_key=os.getenv("MISTRAL_API_KEY"))
class State(TypedDict):
    """
    State for the LangGraph.
    """
    user_input: str
    response: str
    context: str
    urls: list[str]
    llm_choice: str
def check_for_file_node(state: State) -> State:
    if is_file(state["user_input"]):
        # Do file processing
        return {"has_file": True, "files": state["user_input"].files}
    else:
        return {"has_file": False}
def FileConverter(state: State) -> State:
    """
    Check if the user input is a file.
    """
    
    if is_file(state["user_input"]):
        files= state["user_input"].files
        for file in files:
            path= save_uploaded_file(file)
            state["docs"] += docreader(path)
def prompt_doc_merger_node(state: State) -> State:
    """
    Merge the user input with the documents.
    """
    if "docs" in state and state["docs"]:
        state["merged_prompt"] = doc_prompt_merger(state["user_input"], state["docs"])
    else:
        state["merged_prompt"] = state["user_input"]
    return state
def check_for_urls_node(state: State) -> State:
    """
    Check if the user input contains URLs.
    """
    urls = find_urls(state["user_input"])
    if urls:
        state["urls"] = urls
        return {"has_urls": True, "urls": urls}
    else:
        return {"has_urls": False}
def url_type_node(state: State) -> State:
    """
    Classify the URLs into different types.
    """
    if "urls" in state and state["urls"]:
        state["url_types"] = type_url(state["urls"])
    else:
        state["url_types"] = {"youtube": [], "arxiv": [], "web": []}
    return state

def youtube_reader_node(state: State) -> State:
    """
    Read YouTube URLs and return the documents.
    """
    if "url_types" in state and state["url_types"]["youtube"]:
        state["youtube_docs"] = []
        for url in state["url_types"]["youtube"]:
            state["youtube_docs"].extend(youtube_reader(url))
    else:
        state["youtube_docs"] = []
    return state
def web_reader_node(state: State) -> State:
    """    Read web URLs and return the documents.
    """    if "url_types" in state and state["url_types"]["web"]:
        state["web_docs"] = []
        for url in state["url_types"]["web"]:
            state["web_docs"].extend(web_reader(url))
    else:
        state["web_docs"] = []
    return state
def arxiv_reader_node(state: State) -> State:
    """    Read arXiv URLs and return the documents.
    """    if "url_types" in state and state["url_types"]["arxiv"]:
        state["arxiv_docs"] = []
        for url in state["url_types"]["arxiv"]:
            state["arxiv_docs"].extend(arxiv_tool(url))
    else:
        state["arxiv_docs"] = []
    return State
def wikipedia_reader_node(state: State) -> State:
    """    Read Wikipedia based on the user input and return the response.
    """
    if state["user_input"]:
        state["wiki_response"] = wikipedia_tool(state["user_input"])
    else:
        state["wiki_response"] = ""
    return state
def web_reader_tool_node(state: State) -> State:
    """    Read web content based on the user input and return the response.
    """
    if state["user_input"]:
        state["web_response"] = web_tool(state["user_input"])
    else:
        state["web_response"] = ""
    return state
def response_node(state: State) -> State:
    """ Generate the final response based on the context and user input.
    """
    context = state.get("context", "")
    user_input = state["user_input"]
    
    # Combine all the context and user input to form the final response
    response = f"Context: {context}\nUser Input: {user_input}"
    
    # If there are any documents, append them to the response
    if "docs" in state and state["docs"]:
        response += f"\nDocuments: {state['docs']}"
    
    state["response"] = response
    return state
def llm1_node(state: State) -> State:
    """ Generate a response using the first LLM (Cerebras).
    """
    if "merged_prompt" in state:
        response = llm1.invoke(state["merged_prompt"])
        state["response"] = response
    else:
        state["response"] = "No prompt to process."
    return state
def llm2_node(state: State) -> State:
    """ Generate a response using the second LLM (Groq).
    """
    if "merged_prompt" in state:
        response = llm2.invoke(state["merged_prompt"])
        state["response"] = response
    else:
        state["response"] = "No prompt to process."
    return state
def llm3_node(state: State) -> State:
    """ Generate a response using the third LLM (Mistral).
    """
    if "merged_prompt" in state:
        response = llm3.invoke(state["merged_prompt"])
        state["response"] = response
    else:
        state["response"] = "No prompt to process."
    return state 

router_llm=ChatCerebras(model="llama-4-scout-17b-16e-instruct",api_key=os.getenv("CEREBRAS_API_KEY"),temperature=0)
router_llm = ChatOpenAI(model="gpt-4", temperature=0)

routing_prompt = PromptTemplate.from_template("""
You are an intelligent router. Based on the user prompt, choose the most appropriate LLM:

- "groq" for fast and to solve and provide reasoning answers (deepseek-r1-distill-llama-70b)
- "cerebras" for normal queries and answers (llama-4-scout-17b-16e-instruct)
- "mistral" for coding related queries and answers (codestral)

Just reply with one word: groq, cerebras, or mistral.

Prompt: {input}
""")
def routing_node(state: State) -> State:
    """ Route the user input to the appropriate LLM based on the prompt.
    """
    if "user_input" in state:
        response = router_llm.invoke(routing_prompt.invoke({"input": state["user_input"]}))
        state["llm_choice"] = response.strip().lower()
    else:
        state["llm_choice"] = "cerebras"  # Default choice
    return state
def llm_choice_node(state: State) -> State:
    """ Choose the LLM based on the routing decision.
    """
    if "llm_choice" in state:
        if state["llm_choice"] == "groq":
            return llm2_node(state)
        elif state["llm_choice"] == "cerebras":
            return llm1_node(state)
        elif state["llm_choice"] == "mistral":
            return llm3_node(state)
        else:
            state["response"] = "Invalid LLM choice."
    else:
        state["response"] = "No LLM choice made."
    return state
def create_workflow() -> StateGraph:
    """ Create the workflow for the LangGraph.
    """
    return StateGraph(
        nodes=[
            Node("input", user_input_node),
            Node("arxiv_reader", arxiv_reader_node),
            Node("wikipedia_reader", wikipedia_reader_node),
            Node("web_reader", web_reader_tool_node),
            Node("response", response_node),
            Node("llm1", llm1_node),
            Node("llm2", llm2_node),
            Node("llm3", llm3_node),
            Node("routing", routing_node),
            Node("llm_choice", llm_choice_node),
        ]
        edges=[
            Edge(START, "input"),
            Edge("input", "arxiv_reader", condition=lambda state: "arxiv" in state.get("url_types", {})),
            Edge("input", "wikipedia_reader", condition=lambda state: state["user_input"]),
            Edge("input", "web_reader", condition=lambda state: "web" in state.get("url_types", {})),
            Edge("arxiv_reader", "response"),
            Edge("wikipedia_reader", "response"),
            Edge("web_reader", "response"),
            Edge("input", "routing"),
            Edge("routing", "llm_choice"),
            Edge("llm_choice", "llm1", condition=lambda state: state["llm_choice"] == "cerebras"),
            Edge("llm_choice", "llm2", condition=lambda state: state["llm_choice"] == "groq"),
            Edge("llm_choice", "llm3", condition=lambda state: state["llm_choice"] == "mistral"),
            Edge("llm1", END),
            Edge("llm2", END),
            Edge("llm3", END),
        ]
    )
