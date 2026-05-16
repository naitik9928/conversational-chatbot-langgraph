from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from typing import Annotated,TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
import sqlite3
load_dotenv()
conn=sqlite3.connect(database="chatbot_backend.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
model = ChatGroq(model="llama-3.3-70b-versatile")
class ChatState(TypedDict):
    messages:Annotated[list[str],add_messages]

graph=StateGraph(ChatState)
def chat_node(state:ChatState):
    system=SystemMessage(content="You are a helpful ai assistant answer carefully and precise to the cotent!")
    message=[system]+state["messages"]
    response=model.invoke(message)
    return {"messages":[response]}
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
ChatBot=graph.compile(checkpointer=checkpointer)

def retriver():
    store=set()
    for check in checkpointer.list(None):
        store.add(check.config["configurable"]["thread_id"])
    return store
