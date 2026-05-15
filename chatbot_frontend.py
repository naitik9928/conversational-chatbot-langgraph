import streamlit as st
from streamlit_chatbot_backend import ChatBot
from langchain_core.messages import HumanMessage
user_input=st.chat_input("type here")

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
if user_input:
    st.session_state["message_history"].append({"role":"user","message":user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response=ChatBot.invoke({"messages":[HumanMessage(content=user_input)]},config={"configurable":{"thread_id":"1"}})
    with st.chat_message("assistant"):
        st.markdown(response["messages"][-1].content)
    st.session_state["message_history"].append({"role":"assistant","message":response["messages"][-1].content})
