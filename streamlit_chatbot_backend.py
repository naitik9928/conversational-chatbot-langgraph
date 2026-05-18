from dotenv import load_dotenv
import sqlite3

from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import (
    AnyMessage,
    SystemMessage
)


from langgraph.graph import (
    StateGraph,
    START
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.sqlite import SqliteSaver


# =========================================
# LOAD ENV
# =========================================

load_dotenv()


# =========================================
# SEARCH TOOL
# =========================================

from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search_online(query: str):
    """
    Search the internet for latest information.
    """

    response = tavily.search(
        query=query,
        search_depth="basic",
        max_results=5
    )

    return str(response)


@tool
def get_current_date():
    """
    Get the current date and time.
    """

    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================================
# TOOLS
# =========================================

tools = [
    search_online,
    get_current_date
]


# =========================================
# MODEL
# =========================================

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

model_with_tools = model.bind_tools(tools)


# =========================================
# SQLITE MEMORY
# =========================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


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
# CHAT NODE
# =========================================

def chatbot_node(state: ChatState):

    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n"
            "Use tools for:\n"
            "- latest news\n"
            "- current events\n"
            "- today's date\n"
            "- real-time information\n"
        )
    )

    messages = [system_message] + state["messages"]

    response = model_with_tools.invoke(messages)

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
# EDGES
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


# =========================================
# COMPILE GRAPH
# =========================================

ChatBot = graph.compile(
    checkpointer=checkpointer
)


# =========================================
# RETRIEVE THREADS
# =========================================

def retriver():

    store = set()

    try:

        for check in checkpointer.list(None):

            configurable = check.config.get(
                "configurable",
                {}
            )

            thread_id = configurable.get(
                "thread_id"
            )

            if thread_id:

                store.add(thread_id)

    except:
        pass

    return store
