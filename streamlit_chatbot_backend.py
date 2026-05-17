from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from typing import Annotated,TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage,AnyMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
import sqlite3
from langchain_community.tools import DuckDuckGoSearchResults
load_dotenv()
search=DuckDuckGoSearchResults()
@tool
def search_online(query:str):
    """Search the web for latest information and news."""
    response=search.invoke(query)
    return response
@tool
def get_current_date(query:str):
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools=[search_online,get_current_date]
conn=sqlite3.connect(database="chatbot_backend.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
model = ChatGroq(
    model="llama-3.1-8b-instant"
)
model_with_tool=model.bind_tools(tools)
class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

graph=StateGraph(ChatState)
def chat_node(state:ChatState):
    system=SystemMessage(content="You are a helpfull ai assistant answer carefully and precise to the cotent and you can acess tools !")
    message=[system]+state["messages"]
    response=model_with_tool.invoke(message)
    return {"messages":[response]}
tool_node=ToolNode(tools)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_node("tools",tool_node)
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
ChatBot=graph.compile(checkpointer=checkpointer)

def retriver():
    store=set()
    for check in checkpointer.list(None):
        store.add(check.config["configurable"]["thread_id"])
    return store
