import streamlit as st
import uuid

from langchain_core.messages import HumanMessage,AIMessage

from streamlit_chatbot_backend import ChatBot,get_threads,process_pdf


st.set_page_config(
    page_title="LangGraph ChatBot",
    page_icon="🤖",
    layout="centered"
)


def generate_thread_id():

    return str(uuid.uuid4())



def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )



def clear_chat():

    st.session_state["message_history"] = []

    new_thread = generate_thread_id()

    add_thread(new_thread)
    st.session_state["pdf_memory"] = False
    st.session_state["thread_id"] = new_thread


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



if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = list(
        get_threads()
    )
if "pdf_memory" not in st.session_state:
    st.session_state["pdf_memory"] = False

add_thread(
    st.session_state["thread_id"]
)


st.sidebar.title("🤖 LangGraph ChatBot")


if st.sidebar.button("➕ New Chat"):

    clear_chat()


st.sidebar.markdown("---")
st.sidebar.header("Upload PDF Document")
upload_pdf=st.sidebar.file_uploader("Choose a PDF file",type="pdf")
if upload_pdf and not st.session_state["pdf_memory"]:
    temp_path=f"temp_{upload_pdf.name}"
    with open(temp_path,"wb") as f:
        f.write(upload_pdf.read())
    with st.spinner("Processing PDF..."):
        process_pdf(temp_path)
    st.session_state["pdf_memory"] = True
    st.sidebar.success("PDF processed successfully!")




st.sidebar.subheader("Conversations")

for thread in st.session_state["chat_threads"][::-1]:

    thread_name = f"Chat {thread[:8]}"

    if st.sidebar.button(thread_name):

        st.session_state["thread_id"] = thread

        st.session_state["message_history"] = (
            load_conversation(thread)
        )

        st.rerun()




for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])




user_input = st.chat_input(
    "Type your message..."
)



if user_input:

    
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    
    with st.chat_message("user"):

        st.markdown(user_input)


    
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


    
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": final_response
    })
