from dotenv import load_dotenv
import os
import sqlite3
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
retriever_1 = None
def process_pdf(pdf_path: str):
    global retriever_1
    loader=PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(texts, embed)
    retriever_1 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever_1

@tool
def rag_tool(query:str):
    """use this tool to retrieve relevant information from the uploaded PDF document."""
    if retriever_1 is None:
        return "No PDF uploaded. Please upload a PDF to use this tool."
    docs=retriever_1.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context



if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_online(query: str):
    """Search the internet for latest information."""
    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5
        )
        results = response.get("results", [])
        output = ""
        for r in results:
            output += f"Title: {r.get('title')}\n"
            output += f"Content: {r.get('content')}\n\n"
        return output if output else "No results found."
    except Exception as e:
        return f"Search Error: {str(e)}"

@tool
def get_current_date(dummy: str = ""):
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_online, get_current_date,rag_tool]

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    timeout=60
)

model_with_tools = model.bind_tools(tools)

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph = StateGraph(ChatState)

def chatbot_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "Use tools whenever real-time or external information is needed. "
            "Answer clearly and concisely."
        )
    )
    messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

ChatBot = graph.compile(checkpointer=checkpointer)

def get_threads():
    threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            configurable = checkpoint.config.get("configurable", {})
            thread_id = configurable.get("thread_id")
            if thread_id:
                threads.add(thread_id)
    except Exception as e:
        print(f"Retriever Error: {str(e)}")
    return threads

