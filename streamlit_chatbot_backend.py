from dotenv import load_dotenv
import os
import sqlite3

from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI

from langchain.tools import tool

from langchain_core.messages import (
    AnyMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.sqlite import SqliteSaver

from tavily import TavilyClient


# =========================================
# LOAD ENV VARIABLES
# =========================================

load_dotenv()


# =========================================
# CHECK API KEY
# =========================================

if not os.getenv("OPENROUTER_API_KEY"):

    raise ValueError(
        "OPENROUTER_API_KEY not found in .env file"
    )


# =========================================
# TAVILY SEARCH CLIENT
# =========================================

tavily = TavilyClient()


# =========================================
# TOOLS
# =========================================

@tool
def search_online(query: str):
    """
    Search the internet for latest information.
    """

    try:

        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        return str(response)

    except Exception as e:

        return f"Search Error: {str(e)}"


@tool
def get_current_date(dummy: str = ""):
    """
    Get the current date and time.
    """

    from datetime import datetime

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================
# TOOL LIST
# =========================================

tools = [
    search_online,
    get_current_date
]


# =========================================
# MODEL
# =========================================

model = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    timeout=60
)


# =========================================
# BIND TOOLS
# =========================================

model_with_tools = model.bind_tools(tools)


# =========================================
# SQLITE DATABASE
# =========================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)


# =========================================
# CHECKPOINTER
# =========================================

checkpointer = SqliteSaver(conn)

# IMPORTANT
checkpointer.setup()


# =========================================
# STATE
# =========================================

class ChatState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]


# =========================================
# GRAPH
# =========================================

graph = StateGraph(ChatState)


# =========================================
# CHATBOT NODE
# =========================================

def chatbot_node(state: ChatState):

    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n"
            "Use tools whenever real-time or "
            "external information is needed.\n"
            "Answer clearly and concisely."
        )
    )

    messages = [system_message] + state["messages"]

    response = model_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# =========================================
# TOOL NODE
# =========================================

tool_node = ToolNode(tools)


# =========================================
# ADD NODES
# =========================================

graph.add_node(
    "chatbot",
    chatbot_node
)

graph.add_node(
    "tools",
    tool_node
)


# =========================================
# ADD EDGES
# =========================================

graph.add_edge(
    START,
    "chatbot"
)

graph.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph.add_edge(
    "tools",
    "chatbot"
)

graph.add_edge(
    "chatbot",
    END
)


# =========================================
# COMPILE GRAPH
# =========================================

ChatBot = graph.compile(
    checkpointer=checkpointer
)


# =========================================
# RETRIEVE THREADS
# =========================================

def retriever():

    threads = set()

    try:

        for checkpoint in checkpointer.list(None):

            configurable = checkpoint.config.get(
                "configurable",
                {}
            )

            thread_id = configurable.get(
                "thread_id"
            )

            if thread_id:

                threads.add(thread_id)

    except Exception as e:

        print(
            f"Retriever Error: {str(e)}"
        )

    return threads
