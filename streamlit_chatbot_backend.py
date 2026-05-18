from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from typing import Annotated, TypedDict

from langchain_core.messages import (
    SystemMessage,
    AnyMessage,
)

from langchain.tools import tool

from langchain_community.tools import DuckDuckGoSearchResults

import sqlite3


# =====================================
# LOAD ENV
# =====================================

load_dotenv()


# =====================================
# SEARCH TOOL
# =====================================

search = DuckDuckGoSearchResults()


@tool
def search_online(query: str):
    """
    Search the web for latest information and news.
    """

    response = search.invoke(query)

    return response


@tool
def get_current_date():
    """
    Get the current date and time.
    """

    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =====================================
# TOOLS
# =====================================

tools = [
    search_online,
    get_current_date
]


# =====================================
# SQLITE MEMORY
# =====================================

conn = sqlite3.connect(
    database=":memory:",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)


# =====================================
# MODEL
# =====================================

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

model_with_tools = model.bind_tools(tools)


# =====================================
# STATE
# =====================================

class ChatState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]


# =====================================
# GRAPH
# =====================================

graph = StateGraph(ChatState)


# =====================================
# CHAT NODE
# =====================================

def chat_node(state: ChatState):

    system = SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "Answer carefully and precisely. "
            "Use tools whenever needed."
        )
    )

    clean_messages = []

    for msg in state["messages"]:

        if hasattr(msg, "content") and msg.content:

            clean_messages.append(msg)

    messages = [system] + clean_messages

    response = model_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# =====================================
# TOOL NODE
# =====================================

tool_node = ToolNode(tools)


# =====================================
# ADD NODES
# =====================================

graph.add_node("chat_node", chat_node)

graph.add_node("tools", tool_node)


# =====================================
# ADD EDGES
# =====================================

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")


# =====================================
# COMPILE GRAPH
# =====================================

ChatBot = graph.compile(
    checkpointer=checkpointer
)


# =====================================
# RETRIEVE THREADS
# =====================================

def retriver():

    store = set()

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

    return store
