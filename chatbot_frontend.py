from chatbot_groq import retriver,ChatBot
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
import uuid

def thread_generator():
    thread_id=str(uuid.uuid4())
    return thread_id

def clear_chat():
    st.session_state["message_history"]=[]
    thread_id=thread_generator()
    add_thread(thread_id)
    st.session_state["thread_id"]=thread_id
    
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversations(thread_id):
    state=ChatBot.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get("messages",[])

user_input=st.chat_input("type here ")

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=thread_generator()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"]=list(retriver())
add_thread(st.session_state["thread_id"])
st.sidebar.title("LangGraph ChatBot")
if st.sidebar.button("New Chat"):
    clear_chat()
st.sidebar.header("Conversations")
for thread in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread):
        st.session_state["thread_id"]=thread
        messages=load_conversations(thread)
        temp_store=[]
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role="user"
            else:
                role="assistant"
            temp_store.append({"role":role,"messages":msg.content})
        st.session_state["message_history"]=temp_store
        st.rerun()



for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["messages"])

if user_input:
    st.session_state["message_history"].append({"role":"user","messages":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        assistant_text = []

        def ai_only_message():
            for chunk,_ in ChatBot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                config={"configurable":{"thread_id":st.session_state["thread_id"]},"metadata":{"thread_id":st.session_state["thread_id"]},"run_name":"chat_turn"},
                stream_mode="messages"                
            ):
                if isinstance(chunk,AIMessage) and chunk.content and not chunk.tool_calls:
                    assistant_text.append(chunk.content)
                    yield chunk.content

        st.write_stream(ai_only_message())
    st.session_state["message_history"].append({"role":"assistant","messages":"".join(assistant_text)})
