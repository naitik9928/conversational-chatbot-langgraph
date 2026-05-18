from streamlit_chatbot_backend import retriver, ChatBot

import streamlit as st
import uuid

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="LangGraph ChatBot",
    page_icon="🤖",
    layout="centered"
)


# =========================
# THREAD GENERATOR
# =========================

def thread_generator():
    return str(uuid.uuid4())


# =========================
# ADD THREAD
# =========================

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# =========================
# CLEAR CHAT
# =========================

def clear_chat():

    st.session_state["message_history"] = []

    new_thread_id = thread_generator()

    add_thread(new_thread_id)

    st.session_state["thread_id"] = new_thread_id


# =========================
# LOAD CONVERSATION
# =========================

def load_conversation(thread_id):

    state = ChatBot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# =========================
# SESSION STATE
# =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = thread_generator()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(retriver())


# =========================
# ADD CURRENT THREAD
# =========================

add_thread(st.session_state["thread_id"])


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🤖 LangGraph ChatBot")


if st.sidebar.button("➕ New Chat"):
    clear_chat()


st.sidebar.header("Conversations")


for thread in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(thread):

        st.session_state["thread_id"] = thread

        messages = load_conversation(thread)

        temp_store = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                role = "user"

            elif isinstance(msg, AIMessage):

                role = "assistant"

            else:
                continue

            temp_store.append({
                "role": role,
                "messages": msg.content
            })

        st.session_state["message_history"] = temp_store

        st.rerun()


# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["messages"])


# =========================
# CHAT INPUT
# =========================

user_input = st.chat_input("Type your message...")


# =========================
# HANDLE USER INPUT
# =========================

if user_input:

    # Save User Message
    st.session_state["message_history"].append({
        "role": "user",
        "messages": user_input
    })

    # Display User Message
    with st.chat_message("user"):

        st.markdown(user_input)

    # Assistant Response
    with st.chat_message("assistant"):

        def stream_response():

            final_response = ""

            for event in ChatBot.stream(

                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                config={
                    "configurable": {
                        "thread_id": st.session_state["thread_id"]
                    }
                },

                stream_mode="values"

            ):

                messages = event["messages"]

                last_message = messages[-1]

                if (
                    isinstance(last_message, AIMessage)
                    and last_message.content
                    and not last_message.tool_calls
                ):

                    final_response = last_message.content

            yield final_response

        response = st.write_stream(stream_response())

    # Save Assistant Message
    st.session_state["message_history"].append({
        "role": "assistant",
        "messages": response
    })
