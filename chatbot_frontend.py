import streamlit as st
import uuid

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from streamlit_chatbot_backend import (
    ChatBot,
    retriever
)


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="LangGraph ChatBot",
    page_icon="🤖",
    layout="centered"
)


# =========================================
# GENERATE THREAD ID
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

    messages = state.values.get(
        "messages",
        []
    )

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

    return temp_messages


# =========================================
# SESSION STATE
# =========================================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = list(
        retriever()
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


st.sidebar.markdown("---")

st.sidebar.subheader("Conversations")


# =========================================
# SHOW SAVED THREADS
# =========================================

for thread in st.session_state["chat_threads"][::-1]:

    thread_name = f"Chat {thread[:8]}"

    if st.sidebar.button(thread_name):

        st.session_state["thread_id"] = thread

        st.session_state["message_history"] = (
            load_conversation(thread)
        )

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

        response_placeholder = st.empty()

        final_response = ""

        try:

            for chunk, metadata in ChatBot.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },

                config={
                    "configurable": {
                        "thread_id": st.session_state["thread_id"]
                    }
                },

                stream_mode="messages"
            ):

                if isinstance(chunk, AIMessage):

                    if chunk.content:

                        final_response += chunk.content

                        response_placeholder.markdown(
                            final_response
                        )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

            final_response = (
                "Something went wrong."
            )


    # Save assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": final_response
    })
