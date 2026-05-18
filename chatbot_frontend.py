import streamlit as st
import uuid

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from streamlit_chatbot_backend import (
    ChatBot,
    retriver
)


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="LangGraph ChatBot",
    page_icon="🤖"
)


# =========================================
# THREAD GENERATOR
# =========================================

def generate_thread_id():

    return str(uuid.uuid4())


# =========================================
# ADD THREAD
# =========================================

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )


# =========================================
# CLEAR CHAT
# =========================================

def clear_chat():

    st.session_state["message_history"] = []

    new_thread = generate_thread_id()

    add_thread(new_thread)

    st.session_state["thread_id"] = new_thread


# =========================================
# LOAD CONVERSATION
# =========================================

def load_conversation(thread_id):

    state = ChatBot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get(
        "messages",
        []
    )


# =========================================
# SESSION STATE
# =========================================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = list(
        retriver()
    )


# =========================================
# ADD CURRENT THREAD
# =========================================

add_thread(
    st.session_state["thread_id"]
)


# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("🤖 LangGraph ChatBot")


if st.sidebar.button("➕ New Chat"):

    clear_chat()


st.sidebar.header("Conversations")


for thread in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(thread):

        st.session_state["thread_id"] = thread

        messages = load_conversation(thread)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                role = "user"

            elif isinstance(msg, AIMessage):

                role = "assistant"

            else:
                continue

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        st.session_state["message_history"] = temp_messages

        st.rerun()


# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================
# CHAT INPUT
# =========================================

user_input = st.chat_input(
    "Type your message..."
)


# =========================================
# HANDLE USER INPUT
# =========================================

if user_input:

    # Save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):

        response_data = ChatBot.invoke(

            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },

            config={
                "configurable": {
                    "thread_id": st.session_state["thread_id"]
                }
            }
        )

        response = response_data["messages"][-1].content

        st.markdown(response)

    # Save assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": response
    })
