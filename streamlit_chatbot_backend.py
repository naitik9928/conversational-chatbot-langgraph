from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
load_dotenv()
checkpointer=InMemorySaver()
model = ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages:Annotated[list[str],add_messages]

graph=StateGraph(ChatState)



def chat_node(state:ChatState):
    system = SystemMessage(content="You are a helpful AI assistant. Be concise and clear.")
    messages = [system] + state["messages"]
    result = model.invoke(messages)
    return {"messages":[result]}
    

graph.add_node("Chat",chat_node)
graph.add_edge(START,"Chat")
graph.add_edge("Chat",END)

ChatBot=graph.compile(checkpointer=checkpointer)